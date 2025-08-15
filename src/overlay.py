import cv2
import numpy as np
from src.utils import to_int_tuple, draw_text

def draw_skeleton(frame, kp):
    pairs = [
        ("left_shoulder","right_shoulder"),
        ("left_hip","right_hip"),
        ("left_shoulder","left_elbow"), ("left_elbow","left_wrist"),
        ("right_shoulder","right_elbow"), ("right_elbow","right_wrist"),
        ("left_hip","left_knee"), ("left_knee","left_ankle"),
        ("right_hip","right_knee"), ("right_knee","right_ankle"),
        ("left_hip","left_shoulder"), ("right_hip","right_shoulder"),
        ("left_heel","left_foot_index"), ("right_heel","right_foot_index")
    ]
    for a,b in pairs:
        if a in kp and b in kp:
            cv2.line(frame, to_int_tuple(kp[a]), to_int_tuple(kp[b]), (0,255,0), 2)
    for name, pt in kp.items():
        cv2.circle(frame, to_int_tuple(pt), 3, (0,255,255), -1)

def overlay_metrics(frame, m, phase=None, fps=None, y0=24):
    lines = []
    if not np.isnan(m.get("front_elbow_angle_deg", np.nan)):
        lines.append(f"Elbow: {m['front_elbow_angle_deg']:.1f}°")
    if not np.isnan(m.get("spine_lean_deg", np.nan)):
        lines.append(f"Spine lean: {m['spine_lean_deg']:.1f}°")
    if not np.isnan(m.get("head_over_knee_px", np.nan)):
        lines.append(f"Head over knee: {m['head_over_knee_px']:.0f}px")
    if not np.isnan(m.get("front_foot_angle_deg", np.nan)):
        lines.append(f"Front foot: {m['front_foot_angle_deg']:.1f}°")
    if phase:
        lines.append(f"Phase: {phase}")
    if fps is not None:
        lines.append(f"FPS: {fps:.1f}")

    y = y0
    for line in lines:
        draw_text(frame, line, (10, y))
        y += 22

def quick_feedback(frame, m, thresholds, y0=140):
    msgs = []
    e = m.get("front_elbow_angle_deg", float("nan"))
    s = m.get("spine_lean_deg", float("nan"))
    h = m.get("head_over_knee_px", float("nan"))

    e_ok = not np.isnan(e) and thresholds["elbow_min"] <= e <= thresholds["elbow_max"]
    s_ok = not np.isnan(s) and abs(s) <= thresholds["spine_lean_max_deg"]
    h_ok = not np.isnan(h) and h <= thresholds["head_over_knee_max_px"]

    if e_ok: msgs.append("✅ Good elbow elevation")
    if not h_ok and not np.isnan(h): msgs.append("❌ Head not over front knee")
    if not s_ok and not np.isnan(s): msgs.append("❌ Too much side lean")

    y = y0
    for line in msgs:
        draw_text(frame, line, (10, y), scale=0.7, bg=(20,20,200))
        y += 24
