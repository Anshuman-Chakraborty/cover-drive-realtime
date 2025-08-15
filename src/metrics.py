import numpy as np
from src.utils import angle_between_points, line_angle_vs_vertical, point_line_vertical_distance, foot_angle_vs_xaxis

def compute_frame_metrics(kp: dict):
    m = {
        "front_elbow_angle_deg": float("nan"),
        "spine_lean_deg": float("nan"),
        "head_over_knee_px": float("nan"),
        "front_foot_angle_deg": float("nan"),
        "front_side": None
    }
    candidates = []
    for side in ["left", "right"]:
        knee = kp.get(f"{side}_knee")
        if knee is not None:
            candidates.append((side, knee[1]))
    if candidates:
        front_side = sorted(candidates, key=lambda x: x[1])[0][0]
        m["front_side"] = front_side
    else:
        front_side = "left"

    shoulder = kp.get(f"{front_side}_shoulder")
    elbow = kp.get(f"{front_side}_elbow")
    wrist = kp.get(f"{front_side}_wrist")
    if shoulder and elbow and wrist:
        m["front_elbow_angle_deg"] = angle_between_points(shoulder, elbow, wrist)

    shoulders = [kp.get("left_shoulder"), kp.get("right_shoulder")]
    hips = [kp.get("left_hip"), kp.get("right_hip")]
    if all(shoulders) and all(hips):
        sh = np.mean(np.array(shoulders), axis=0)
        hp = np.mean(np.array(hips), axis=0)
        m["spine_lean_deg"] = line_angle_vs_vertical(hp, sh)

    nose = kp.get("nose")
    knee_f = kp.get(f"{front_side}_knee")
    if nose and knee_f:
        m["head_over_knee_px"] = point_line_vertical_distance(nose[0], knee_f[0])

    heel = kp.get(f"{front_side}_heel")
    toe = kp.get(f"{front_side}_foot_index")
    if heel and toe:
        m["front_foot_angle_deg"] = foot_angle_vs_xaxis(heel, toe)

    return m
