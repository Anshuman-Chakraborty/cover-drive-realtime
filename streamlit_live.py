import os, time, traceback, tempfile, sys
from pathlib import Path

import cv2, numpy as np, pandas as pd, streamlit as st

# Ensure src is importable when executing from project root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pose_estimator import PoseEstimator
from metrics import compute_frame_metrics
from overlay import draw_skeleton, overlay_metrics, quick_feedback
from utils import ensure_dir, fps_smoother

try:
    import yt_dlp as ytdlp
except Exception:
    ytdlp = None

st.set_page_config(page_title="Cover Drive Live Analyzer", layout="wide")
st.title("🏏 Cover Drive — Live Analysis (1s updates)")

# Sidebar controls
with st.sidebar:
    st.header("Input")
    source = st.radio("Source", ["YouTube URL", "Upload file"], index=0)
    target_fps = st.slider("Target FPS (processing)", 8, 30, 16)
    max_width = st.slider("Max processing width (px)", 480, 960, 640, step=20)
    run_btn = st.button("Start / Restart Analysis", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("CSV log and annotated video are saved to `./output`")

# Placeholders & layout
status_box = st.empty()
left_col, right_col = st.columns([2,1])
video_frame_ph = left_col.empty()
metrics_cols = right_col.columns(4)
chart_elbow_ph = st.empty()
chart_spine_ph = st.empty()
chart_head_ph = st.empty()

OUTPUT_DIR = "./output"
ensure_dir(OUTPUT_DIR)

# session state
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=["t_sec","elbow_deg","spine_deg","head_over_knee_px","foot_deg"])
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "last_plot" not in st.session_state:
    st.session_state.last_plot = 0.0

def download_if_url(src: str, out_path: str) -> str:
    if os.path.exists(src):
        return src
    if src.startswith("http"):
        if ytdlp is None:
            raise RuntimeError("yt-dlp not installed; cannot download from URL.")
        ydl_opts = {'outtmpl': out_path, 'format': 'mp4/bestaudio/best', 'quiet': True, 'no_warnings': True}
        with ytdlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([src])
        return out_path
    raise FileNotFoundError(f"Input not found: {src}")

def open_capture(input_value: str):
    local = os.path.join(OUTPUT_DIR, "input_live.mp4")
    path = download_if_url(input_value, local)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video capture.")
    return cap

def log_csv_row(path, row):
    is_new = not os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, mode='a', index=False, header=is_new)

def render_metric_cards(m):
    vals = [
        ("Elbow°", m.get("front_elbow_angle_deg", float("nan"))),
        ("Spine°", m.get("spine_lean_deg", float("nan"))),
        ("Head over knee px", m.get("head_over_knee_px", float("nan"))),
        ("Front foot°", m.get("front_foot_angle_deg", float("nan"))),
    ]
    for col, (label, val) in zip(metrics_cols, vals):
        with col:
            if pd.isna(val):
                st.metric(label, "—")
            else:
                st.metric(label, f"{val:.1f}")

def process_stream(input_value: str, target_fps: int, max_width: int):
    # reset state
    st.session_state.df = st.session_state.df.iloc[0:0]
    st.session_state.start_time = time.time()
    st.session_state.last_plot = 0.0

    log_path = os.path.join(OUTPUT_DIR, "live_session_log.csv")
    try:
        if os.path.exists(log_path):
            os.remove(log_path)
    except Exception:
        pass

    status_box.info("Preparing input...")
    cap = open_capture(input_value)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(round(orig_fps / target_fps))) if target_fps > 0 else 1

    pose = PoseEstimator(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5, smooth_landmarks=True)

    smoothed_fps = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    annotated_path = os.path.join(OUTPUT_DIR, "live_annotated.mp4")
    writer = None
    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_interval != 0:
            i += 1
            continue

        # resize
        h0, w0 = frame.shape[:2]
        if w0 > max_width:
            scale = max_width / float(w0)
            frame = cv2.resize(frame, (max_width, int(h0 * scale)) , interpolation=cv2.INTER_AREA)

        h, w = frame.shape[:2]
        t1 = time.time()
        results = pose.infer(frame)
        kp = pose.extract_keypoints(results, w, h)
        m = compute_frame_metrics(kp)

        if writer is None:
            writer = cv2.VideoWriter(annotated_path, fourcc, float(target_fps), (w, h))

        draw_skeleton(frame, kp)
        smoothed_fps = fps_smoother(smoothed_fps, 1.0 / max(time.time() - t1, 1e-6), alpha=0.85)
        overlay_metrics(frame, m, phase=None, fps=smoothed_fps)
        quick_feedback(frame, m, {
            "elbow_min": 100, "elbow_max": 160,
            "spine_lean_max_deg": 20,
            "head_over_knee_max_px": 30,
        })

        # write annotated
        writer.write(frame)

        # display frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame_ph.image(frame_rgb, channels="RGB", use_container_width=True)

        # metrics
        render_metric_cards(m)

        # append row
        t_sec = round(time.time() - st.session_state.start_time, 1)
        row = {"t_sec": t_sec,
               "elbow_deg": float(m.get("front_elbow_angle_deg", float("nan"))),
               "spine_deg": float(m.get("spine_lean_deg", float("nan"))),
               "head_over_knee_px": float(m.get("head_over_knee_px", float("nan"))),
               "foot_deg": float(m.get("front_foot_angle_deg", float("nan"))),
               }
        st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([row])], ignore_index=True)

        # log once per second (nearest)
        if int(t_sec) != int(st.session_state.last_plot):
            try:
                log_csv_row(log_path, row)
            except Exception:
                pass

        # update charts 1 Hz
        if time.time() - st.session_state.last_plot >= 1.0:
            dfp = st.session_state.df.copy()
            if not dfp.empty:
                chart_elbow_ph.line_chart(dfp.set_index("t_sec")[["elbow_deg"]])
                chart_spine_ph.line_chart(dfp.set_index("t_sec")[["spine_deg"]])
                chart_head_ph.line_chart(dfp.set_index("t_sec")[["head_over_knee_px"]])
            st.session_state.last_plot = time.time()

        i += 1

    cap.release()
    if writer is not None:
        writer.release()

    status_box.success(f"Finished. Saved log: {log_path} and annotated video: {annotated_path}")

# UI behavior
if run_btn:
    try:
        if source == "YouTube URL":
            url = st.text_input("YouTube URL", placeholder="https://youtube.com/shorts/...")
            if not url:
                st.warning("Paste a YouTube URL in the text box then click Start.")
            else:
                process_stream(url, target_fps, max_width)
        else:
            up = st.file_uploader("Upload a video file", type=["mp4","mov","avi","mkv"])
            if up is None:
                st.warning("Upload a file then click Start.")
            else:
                # write temp file and process
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(up.read())
                    tmp.flush()
                    process_stream(tmp.name, target_fps, max_width)
    except Exception as e:
        st.error(f"Processing failed: {e}")
        st.code(traceback.format_exc())
