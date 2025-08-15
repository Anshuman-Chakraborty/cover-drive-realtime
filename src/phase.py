import numpy as np

def segment_phases(metrics_log, window=5):
    if not metrics_log:
        return [], None

    elbow = np.array([m.get("front_elbow_angle_deg", np.nan) for m in metrics_log], dtype=float)
    foot  = np.array([m.get("front_foot_angle_deg", np.nan) for m in metrics_log], dtype=float)

    elbow = np.nan_to_num(elbow, nan=np.nanmedian(elbow) if not np.isnan(np.nanmedian(elbow)) else 120.0)
    foot  = np.nan_to_num(foot,  nan=np.nanmedian(foot)  if not np.isnan(np.nanmedian(foot))  else 20.0)

    vel_elbow = np.gradient(elbow)
    vel_foot  = np.gradient(foot)

    phases = []
    impact_idx = int(np.argmax(np.abs(vel_elbow))) if len(vel_elbow)>0 else 0

    for i in range(len(metrics_log)):
        if i < impact_idx - 10:
            if abs(vel_foot[i]) < 0.5 and abs(vel_elbow[i]) < 0.5:
                phases.append("Stance/Address")
            elif vel_foot[i] > 0.5:
                phases.append("Stride")
            else:
                phases.append("Downswing Prep")
        elif abs(i - impact_idx) <= 3:
            phases.append("Impact")
        elif i <= impact_idx + 12:
            phases.append("Follow-through")
        else:
            phases.append("Recovery")
    return phases, impact_idx
