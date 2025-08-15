import mediapipe as mp

class PoseEstimator:
    def __init__(self, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5, smooth_landmarks=True):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            smooth_landmarks=smooth_landmarks
        )

    def infer(self, image_bgr):
        image_rgb = image_bgr[:, :, ::-1]
        results = self.pose.process(image_rgb)
        return results

    @staticmethod
    def extract_keypoints(results, width, height):
        lm = getattr(results, "pose_landmarks", None)
        if not lm:
            return {}
        lm = lm.landmark
        idx = {
            "nose": 0,
            "left_shoulder": 11, "right_shoulder": 12,
            "left_elbow": 13, "right_elbow": 14,
            "left_wrist": 15, "right_wrist": 16,
            "left_hip": 23, "right_hip": 24,
            "left_knee": 25, "right_knee": 26,
            "left_ankle": 27, "right_ankle": 28,
            "left_heel": 29, "right_heel": 30,
            "left_foot_index": 31, "right_foot_index": 32
        }
        out = {}
        for name, i in idx.items():
            if i < len(lm):
                li = lm[i]
                if getattr(li, "visibility", 0.0) > 0.2:
                    out[name] = (li.x * width, li.y * height)
        return out
