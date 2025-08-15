import os, time, json, yaml, traceback
import cv2
import numpy as np
from tqdm import tqdm

from src.pose_estimator import PoseEstimator
from src.metrics import compute_frame_metrics
from src.overlay import draw_skeleton, overlay_metrics, quick_feedback
from src.phase import segment_phases
from src.evaluator import final_evaluation
from src.utils import ensure_dir, fps_smoother
from src.report import write_report

try:
    import yt_dlp as ytdlp
except Exception:
    ytdlp = None

def download_video_if_needed(input_path_or_url, out_path):
    if os.path.exists(input_path_or_url):
        return input_path_or_url
    if input_path_or_url.startswith("http"):
        if ytdlp is None:
            raise RuntimeError("yt-dlp not installed; cannot download from URL.")
        ydl_opts = {'outtmpl': out_path, 'format': 'mp4/bestaudio/best'}
        with ytdlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([input_path_or_url])
        return out_path
    raise FileNotFoundError(f"{input_path_or_url} not found")

def analyze_video(input_path_or_url: str, config_path: str = "./config.yaml") -> dict:
    import matplotlib
    matplotlib.use("Agg")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["paths"]["output_dir"]
    reports_dir = cfg["paths"]["reports_dir"]
    ensure_dir(output_dir); ensure_dir(reports_dir)

    local_in = os.path.join(output_dir, "input.mp4")
    src_path = download_video_if_needed(input_path_or_url, local_in)

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    target_fps = cfg["video"]["target_fps"] or int(orig_fps)
    frame_interval = max(1, int(round(orig_fps / target_fps))) if target_fps > 0 else 1

    fourcc = cv2.VideoWriter_fourcc(*cfg["video"]["fourcc"])

    pose = PoseEstimator(**cfg["pose"])

    annotated_path = os.path.join(output_dir, "annotated_video.mp4")
    out_writer = None
    metrics_log = []
    phase_labels = None
    impact_idx = None

    t0 = time.time()
    smoothed_fps = None
    frame_count_out = 0
    frame_count_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    pbar = tqdm(total=frame_count_in if frame_count_in>0 else None, desc="Processing", unit="frame")

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if i % frame_interval != 0:
            i += 1; pbar.update(1); continue

        frame = frame
        h, w = frame.shape[:2]
        t1 = time.time()
        results = pose.infer(frame)
        kp = pose.extract_keypoints(results, w, h)
        m = compute_frame_metrics(kp)
        metrics_log.append(m)

        if out_writer is None:
            out_writer = cv2.VideoWriter(annotated_path, fourcc, float(target_fps), (w, h))

        draw_skeleton(frame, kp)

        if len(metrics_log) == 30 and phase_labels is None:
            phase_labels, impact_idx = segment_phases(metrics_log)

        phase = phase_labels[len(metrics_log)-1] if phase_labels and len(phase_labels)>=len(metrics_log) else None

        inst_fps = 1.0 / max(time.time() - t1, 1e-6)
        smoothed_fps = fps_smoother(smoothed_fps, inst_fps, alpha=0.85)

        overlay_metrics(frame, m, phase=phase, fps=smoothed_fps)
        quick_feedback(frame, m, cfg["metrics_thresholds"])

        out_writer.write(frame)
        frame_count_out += 1
        i += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if out_writer is not None:
        out_writer.release()

    evaluation = final_evaluation(metrics_log, cfg["reference_ranges"], cfg["scoring_weights"])

    eval_path = os.path.join(output_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(evaluation, f, indent=2)

    try:
        import matplotlib.pyplot as plt
        elbow = [m.get("front_elbow_angle_deg", np.nan) for m in metrics_log]
        spine = [m.get("spine_lean_deg", np.nan) for m in metrics_log]
        hok   = [m.get("head_over_knee_px", np.nan) for m in metrics_log]

        plt.figure(); plt.plot(elbow); plt.xlabel("Frame"); plt.ylabel("Elbow angle (deg)"); plt.title("Elbow angle over time")
        plt.savefig(os.path.join(output_dir, "elbow_angle.png"), dpi=150); plt.close()

        plt.figure(); plt.plot(spine); plt.xlabel("Frame"); plt.ylabel("Spine lean vs vertical (deg)"); plt.title("Spine lean over time")
        plt.savefig(os.path.join(output_dir, "spine_lean.png"), dpi=150); plt.close()

        plt.figure(); plt.plot(hok); plt.xlabel("Frame"); plt.ylabel("Head-over-knee (px)"); plt.title("Head over knee over time")
        plt.savefig(os.path.join(output_dir, "head_over_knee.png"), dpi=150); plt.close()
    except Exception:
        pass

    meta = {"video_path": src_path, "frames": frame_count_out, "fps": (frame_count_out / max(time.time() - t0, 1e-6))}
    report_path = write_report(reports_dir, evaluation, meta, include_plots=True)

    return {"annotated_video": annotated_path, "evaluation_json": eval_path, "report_html": report_path, "meta": meta}

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(description="Real-Time Cover Drive Analysis")
    parser.add_argument("--input", required=True, help="Path or URL to input video (e.g., YouTube short URL)")
    parser.add_argument("--config", default="./config.yaml")
    args = parser.parse_args()
    try:
        out = analyze_video(args.input, args.config)
        print(json.dumps(out, indent=2))
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
