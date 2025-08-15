import os
import cv2
import numpy as np
from math import atan2, degrees

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def to_int_tuple(pt):
    return (int(pt[0]), int(pt[1]))

def angle_between_points(a, b, c):
    """
    Returns angle ABC (in degrees) where B is the vertex.
    """
    try:
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cosang = np.clip(cosang, -1.0, 1.0)
        ang = np.degrees(np.arccos(cosang))
        return float(ang)
    except Exception:
        return float('nan')

def line_angle_vs_vertical(p1, p2):
    """
    Angle of line p1->p2 relative to vertical axis (degrees).
    0 means perfectly vertical; positive when leaning to the right.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    theta_h = degrees(atan2(dy, dx))
    theta_v = 90 - theta_h
    while theta_v > 180: theta_v -= 360
    while theta_v < -180: theta_v += 360
    return float(theta_v)

def point_line_vertical_distance(px, knee_x):
    return abs(px - knee_x)

def foot_angle_vs_xaxis(heel, toe):
    dx = toe[0] - heel[0]
    dy = toe[1] - heel[1]
    return degrees(atan2(dy, dx))

def draw_text(frame, text, org, scale=0.6, thickness=2, color=(255,255,255), bg=(0,0,0)):
    x, y = int(org[0]), int(org[1])
    (w, h), baseline = cv2.getTextSize(str(text), cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(frame, (x, y - h - baseline), (x + w, y + baseline), bg, -1)
    cv2.putText(frame, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def fps_smoother(prev_fps, inst_fps, alpha=0.9):
    if prev_fps is None: return inst_fps
    return alpha * prev_fps + (1 - alpha) * inst_fps
