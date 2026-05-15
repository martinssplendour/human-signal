import cv2
import numpy as np


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 13, 14, 78, 308]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
PNP_LANDMARKS = [1, 152, 33, 263, 61, 291]
PNP_MODEL_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, -330.0, -65.0],
        [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0],
        [-150.0, -150.0, -125.0],
        [150.0, -150.0, -125.0],
    ],
    dtype=np.float64,
)


def _aspect_ratio(pts):
    p = pts
    vert = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horiz = np.linalg.norm(p[0] - p[3]) + 1e-6
    return vert / (2.0 * horiz)


def eye_aspect_ratio(face):
    if face is None:
        return None, None
    le = face[LEFT_EYE, :2]
    re = face[RIGHT_EYE, :2]
    return _aspect_ratio(le), _aspect_ratio(re)


def mouth_aspect_ratio(face):
    if face is None:
        return None
    m = face[MOUTH, :2]
    vert = np.linalg.norm(m[2] - m[3])
    horiz = np.linalg.norm(m[0] - m[1]) + 1e-6
    return vert / horiz


def iris_gaze(face):
    if face is None or len(face) < 478:
        return {"x": 0.0, "y": 0.0, "score": 0.0, "confidence": 0.0}

    def _eye_offset(corner_a, corner_b, top_idxs, bottom_idxs, iris_idxs):
        a = face[corner_a, :2]
        b = face[corner_b, :2]
        iris = face[np.array(iris_idxs), :2].mean(axis=0)
        width = np.linalg.norm(b - a) + 1e-6
        top = face[np.array(top_idxs), :2].mean(axis=0)
        bottom = face[np.array(bottom_idxs), :2].mean(axis=0)
        height = np.linalg.norm(bottom - top) + 1e-6
        center = (a + b) / 2.0
        return np.array([(iris[0] - center[0]) / width, (iris[1] - center[1]) / height], dtype=np.float32)

    left = _eye_offset(33, 133, [159, 158, 160], [145, 153, 144], LEFT_IRIS)
    right = _eye_offset(362, 263, [386, 385, 387], [374, 373, 380], RIGHT_IRIS)
    offset = (left + right) / 2.0
    magnitude = float(np.linalg.norm(offset))
    score = float(np.clip(1.0 - magnitude / 0.22, 0.0, 1.0))
    return {
        "x": float(offset[0]),
        "y": float(offset[1]),
        "score": score,
        "confidence": 1.0,
    }


def gaze_proxy(face):
    return iris_gaze(face)["score"]


def neck_angle_deg(pose):
    if pose is None:
        return None
    try:
        l_sh, r_sh = pose[11, :2], pose[12, :2]
        l_ear, r_ear = pose[7, :2], pose[8, :2]
        shoulder_mid = (l_sh + r_sh) / 2.0
        ear_mid = (l_ear + r_ear) / 2.0
        v = ear_mid - shoulder_mid
        v_norm = v / (np.linalg.norm(v) + 1e-6)
        dot = np.dot(v_norm, np.array([0.0, -1.0]))
        return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))
    except Exception:
        return None


def legacy_head_pose_proxy(face):
    if face is None or len(face) < 264:
        return {"yaw": None, "pitch": None, "face_area": None}
    xs, ys = face[:, 0], face[:, 1]
    face_w = xs.max() - xs.min() + 1e-6
    face_h = ys.max() - ys.min() + 1e-6
    nose = face[1, :2]
    left_center = face[np.array(LEFT_EYE), :2].mean(axis=0)
    right_center = face[np.array(RIGHT_EYE), :2].mean(axis=0)
    eye_mid = (left_center + right_center) / 2.0
    face_mid = np.array([xs.mean(), ys.mean()])
    return {
        "yaw": float((nose[0] - face_mid[0]) / face_w),
        "pitch": float((nose[1] - eye_mid[1]) / face_h),
        "face_area": float(face_w * face_h),
    }


def head_pose_proxy(face, frame_size=None):
    legacy = legacy_head_pose_proxy(face)
    if face is None or len(face) <= max(PNP_LANDMARKS):
        return {**legacy, "yaw_deg": None, "pitch_deg": None, "roll_deg": None}
    try:
        face_2d = face[np.array(PNP_LANDMARKS), :2].astype(np.float64)
        if frame_size is None:
            xs, ys = face[:, 0], face[:, 1]
            width = float(max(xs.max() + 1.0, np.ptp(xs) + 1.0))
            height = float(max(ys.max() + 1.0, np.ptp(ys) + 1.0))
        else:
            width, height = frame_size
        focal_length = width
        camera_matrix = np.array(
            [[focal_length, 0.0, width / 2.0], [0.0, focal_length, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(PNP_MODEL_POINTS, face_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return {**legacy, "yaw_deg": None, "pitch_deg": None, "roll_deg": None}
        rmat, _ = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        pitch_deg, yaw_deg, roll_deg = [float(a) for a in angles]
        return {**legacy, "yaw_deg": yaw_deg, "pitch_deg": pitch_deg, "roll_deg": roll_deg}
    except Exception:
        return {**legacy, "yaw_deg": None, "pitch_deg": None, "roll_deg": None}


def head_pose_from_transform(matrix):
    if matrix is None:
        return {"yaw_deg": None, "pitch_deg": None, "roll_deg": None}
    try:
        rot = np.asarray(matrix, dtype=np.float32)[:3, :3]
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        pitch = np.arctan2(-rot[2, 0], np.sqrt(rot[2, 1] ** 2 + rot[2, 2] ** 2))
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        return {
            "yaw_deg": float(np.degrees(yaw)),
            "pitch_deg": float(np.degrees(pitch)),
            "roll_deg": float(np.degrees(roll)),
        }
    except Exception:
        return {"yaw_deg": None, "pitch_deg": None, "roll_deg": None}


def blendshape_value(blendshapes, *names):
    if not blendshapes:
        return None
    lowered = {str(k).lower(): v for k, v in blendshapes.items()}
    values = []
    for name in names:
        value = lowered.get(name.lower())
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    return float(max(values))


def compute_features(det):
    face = det["face"]
    pose = det["pose"]
    le, re = eye_aspect_ratio(face)
    mar = mouth_aspect_ratio(face)
    gaze_info = iris_gaze(face)
    neck = neck_angle_deg(pose)
    head = head_pose_proxy(face, det.get("size"))
    transform_head = head_pose_from_transform(det.get("face_transform"))
    blendshapes = det.get("blendshapes", {})
    eye_closed_score = blendshape_value(blendshapes, "eyeBlinkLeft", "eyeBlinkRight")
    mouth_open_score = blendshape_value(blendshapes, "jawOpen", "mouthFunnel", "mouthPucker")
    head_yaw_deg = transform_head["yaw_deg"] if transform_head["yaw_deg"] is not None else head["yaw_deg"]
    head_pitch_deg = transform_head["pitch_deg"] if transform_head["pitch_deg"] is not None else head["pitch_deg"]
    head_roll_deg = transform_head["roll_deg"] if transform_head["roll_deg"] is not None else head["roll_deg"]
    return {
        "face": face,
        "pose": pose,
        "ear_left": le,
        "ear_right": re,
        "mar": mar,
        "gaze": gaze_info["score"],
        "gaze_x": gaze_info["x"],
        "gaze_y": gaze_info["y"],
        "gaze_confidence": gaze_info["confidence"],
        "gaze_vector": gaze_info,
        "neck_angle": neck,
        "head_yaw": head["yaw"],
        "head_pitch": head["pitch"],
        "head_yaw_deg": head_yaw_deg,
        "head_pitch_deg": head_pitch_deg,
        "head_roll_deg": head_roll_deg,
        "head_roll": head_roll_deg,
        "face_area": head["face_area"],
        "eye_closed_score": eye_closed_score,
        "mouth_open_score": mouth_open_score,
        "blendshapes": blendshapes,
        "tracker_backend": det.get("tracker_backend"),
    }
