import numpy as np

def score_category(series, target_range, invert=False):
    s = np.array(series, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) == 0:
        return 5.0
    low, high = target_range
    within = (s >= low) & (s <= high)
    frac = within.mean()
    return 1.0 + 9.0 * float(frac)

def overall_grade(score):
    if score >= 8.5: return "Advanced"
    if score >= 6.5: return "Intermediate"
    return "Beginner"

def final_evaluation(metrics_log, reference, weights):
    elbow = [m.get("front_elbow_angle_deg", np.nan) for m in metrics_log]
    spine = [m.get("spine_lean_deg", np.nan) for m in metrics_log]
    hok  = [m.get("head_over_knee_px", np.nan) for m in metrics_log]
    foot = [m.get("front_foot_angle_deg", np.nan) for m in metrics_log]

    scores = {}
    scores["Footwork"] = score_category(foot, reference.get("front_foot_deg",[10,45]))
    scores["Head Position"] = score_category(hok, reference.get("head_over_knee_px",[0,20]), invert=True)
    scores["Swing Control"] = score_category(elbow, reference.get("elbow_deg",[110,150]))
    scores["Balance"] = score_category(spine, reference.get("spine_lean_deg",[-15,10]))
    scores["Follow-through"] = np.mean([scores["Swing Control"], scores["Balance"]])

    overall = 0.0
    denom = 0.0
    for k,v in scores.items():
        w = float(weights.get(k, 0.2))
        overall += w * v
        denom += w
    overall = overall / max(denom, 1e-6)
    grade = overall_grade(overall)

    feedback = {
        "Footwork": "Aim for a comfortable open stance with the front foot pointing slightly towards cover (10–45°).",
        "Head Position": "Keep the head stacked over the front knee at impact to stay balanced and transfer weight forward.",
        "Swing Control": "Maintain a firm but relaxed front elbow (≈110–150°) for a high elbow and straight bat path.",
        "Balance": "Minimize side lean; a slight forward lean (−15° to 10°) helps getting over the ball.",
        "Follow-through": "Finish high and controlled; let the hands extend after contact without losing shape."
    }

    return {"scores": scores, "overall": overall, "grade": grade, "feedback": feedback}
