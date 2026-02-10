import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# OpenPose python wrapper path
sys.path.append(r"C:\openpose_build\openpose\build\python\openpose\Release")
import pyopenpose as op

# ==============================
# SETTINGS
# ==============================
COCO_MODEL = "yolov8n.pt"     # COCO person detector
SPATULA_MODEL = "best.pt"     # your custom model (spatula detector)
SPATULA_CLASS_ID = 1          # in your best.pt: 1 = spatula
SPATULA_LENGTH_CM = 15.0

OPENPOSE_MODELS = r"C:\openpose_build\openpose\models"

RESIZE_WIDTH = 640            # keep aspect ratio
BBOX_MARGIN = 0.05            # bbox expansion
BBOX_TOP_CROP = 0.06          # remove top part to avoid sheet edges

KP_CONF_THR = 0.03            # OpenPose keypoint confidence threshold

# BODY_25 indices
MID_HIP = 8

# Right leg
R_HIP, R_KNEE, R_ANKLE = 9, 10, 11
R_HEEL = 24

# Left leg
L_HIP, L_KNEE, L_ANKLE = 12, 13, 14
L_HEEL = 21

# Face/Head anchor keypoints (BODY_25)
NOSE, NECK = 0, 1
REYE, LEYE, REAR, LEAR = 15, 16, 17, 18


# ==============================
# HELPERS
# ==============================
def kp_ok(kp, thr=KP_CONF_THR):
    return kp is not None and float(kp[2]) > thr


def dist2d(a, b):
    return float(np.linalg.norm(
        np.array([a[0], a[1]], dtype=np.float32) -
        np.array([b[0], b[1]], dtype=np.float32))
    )


def clamp_bbox(b, w, h):
    x1, y1, x2, y2 = b
    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(0, min(w - 1, x2)))
    y2 = int(max(0, min(h - 1, y2)))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return (x1, y1, x2, y2)


def add_margin_bbox(b, w, h, margin=BBOX_MARGIN):
    x1, y1, x2, y2 = b
    bw, bh = (x2 - x1), (y2 - y1)
    mx, my = int(bw * margin), int(bh * margin)
    return clamp_bbox((x1 - mx, y1 - my, x2 + mx, y2 + my), w, h)


def crop_bbox_top(b, frac=BBOX_TOP_CROP):
    if frac <= 0:
        return b
    x1, y1, x2, y2 = b
    bh = (y2 - y1)
    y1 = int(y1 + bh * frac)
    return (x1, y1, x2, y2)


def draw_dot(img, x, y, color, r=10, text=None):
    x, y = int(x), int(y)
    cv2.circle(img, (x, y), r + 6, (0, 0, 0), -1)
    cv2.circle(img, (x, y), r, color, -1)
    if text is not None:
        cv2.putText(img, str(text), (x + r + 6, y - r - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 6)
        cv2.putText(img, str(text), (x + r + 6, y - r - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def draw_keypoints(img, kps, color=(0, 255, 0), conf_thr=KP_CONF_THR):
    if kps is None:
        return
    for i in range(kps.shape[0]):
        x, y, c = kps[i]
        if float(c) <= conf_thr:
            continue
        draw_dot(img, x, y, color, r=10, text=i)


def draw_line(img, p1, p2, color=(255, 255, 0), thick=5):
    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))
    cv2.line(img, p1, p2, (0, 0, 0), thick + 6)
    cv2.line(img, p1, p2, color, thick)


def pick_best_box_by_area(results, target_cls, conf_min=0.15):
    best = None
    best_score = -1.0
    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box
        if int(cls) != int(target_cls):
            continue
        conf = float(conf)
        if conf < conf_min:
            continue
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        area = max(1, x2 - x1) * max(1, y2 - y1)
        score = area * conf
        if score > best_score:
            best_score = score
            best = (x1, y1, x2, y2, conf)
    return best


# ==============================
# HEAD: anchor + robust top-of-head
# ==============================
def get_head_anchor_from_pose(kps, bbox):
    bx1, by1, bx2, by2 = bbox
    head_ids = [NOSE, REYE, LEYE, REAR, LEAR]

    pts = []
    for idx in head_ids:
        if idx < kps.shape[0] and kp_ok(kps[idx]):
            pts.append((float(kps[idx][0]), float(kps[idx][1])))

    if pts:
        ax = int(np.mean([p[0] for p in pts]))
        ay = int(min([p[1] for p in pts]))
        return ax, ay

    if NECK < kps.shape[0] and kp_ok(kps[NECK]):
        ax = int(kps[NECK][0])
        ay = int(kps[NECK][1] - 0.12 * (by2 - by1))
        ay = max(by1, ay)
        return ax, ay

    return int((bx1 + bx2) / 2), int(by1 + 0.12 * (by2 - by1))


def find_top_of_head_edge(img_bgr, bbox, anchor_xy):
    bx1, by1, bx2, by2 = bbox
    h, w = img_bgr.shape[:2]
    bx1, by1, bx2, by2 = clamp_bbox((bx1, by1, bx2, by2), w, h)

    ax, ay = anchor_xy
    ax = int(np.clip(ax, bx1, bx2))
    ay = int(np.clip(ay, by1, by2))

    roi = img_bgr[by1:by2, bx1:bx2].copy()
    rh, rw = roi.shape[:2]
    ax_r = ax - bx1
    ay_r = ay - by1

    y_end = max(5, min(ay_r, int(rh * 0.70)))

    half_win = max(14, int(rw * 0.05))
    xL = max(0, ax_r - half_win)
    xR = min(rw - 1, ax_r + half_win)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    window_w = max(1, xR - xL)
    min_energy_pixels = max(8, int(window_w * 0.12))
    thresh = 35

    for yy in range(0, y_end):
        row = mag[yy, xL:xR]
        strong = np.count_nonzero(row > thresh)
        if strong >= min_energy_pixels:
            xs = np.where(row > thresh)[0]
            x_mean = int(np.mean(xs)) + xL
            return (bx1 + x_mean, by1 + yy)

    return (ax, by1)


# ==============================
# LEG LENGTH (HEEL ONLY) + MAX(L,R)
# ==============================
def leg_length_polyline_to_heel(kps, hip_i, knee_i, ankle_i, heel_i):
    """
    hip->knee->ankle in 2D + ankle->heel (ONLY heel, no toes).
    Returns (leg_px, heel_point_or_None)
    """
    if not (kp_ok(kps[hip_i]) and kp_ok(kps[knee_i]) and kp_ok(kps[ankle_i])):
        return None, None

    hip = kps[hip_i]
    knee = kps[knee_i]
    ankle = kps[ankle_i]

    leg_px = dist2d(hip, knee) + dist2d(knee, ankle)

    heel_pt = None
    if heel_i < kps.shape[0] and kp_ok(kps[heel_i]):
        heel_pt = kps[heel_i]
        leg_px += dist2d(ankle, heel_pt)

    return leg_px, heel_pt


# ==============================
# MAIN
# ==============================
def cal_height(image_path: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    img0 = cv2.imread(image_path)
    if img0 is None:
        print(f"❌ Could not read image: {image_path}")
        return False

    oh, ow = img0.shape[:2]
    print(f"Original image size: {ow}x{oh}")

    scale = RESIZE_WIDTH / float(ow)
    new_h = int(oh * scale)
    img = cv2.resize(img0, (RESIZE_WIDTH, new_h), interpolation=cv2.INTER_AREA)

    name_root, ext = os.path.splitext(os.path.basename(image_path))
    resized_path = os.path.join(os.path.dirname(image_path), f"{name_root}_resized{ext}")
    cv2.imwrite(resized_path, img)
    print(f"Resized image saved to: {resized_path}")
    print(f"New size: {RESIZE_WIDTH}x{new_h}")

    H, W = img.shape[:2]

    print("Loading YOLO models (COCO person + custom spatula)...")
    coco_model = YOLO(os.path.join(base_dir, COCO_MODEL))
    spatula_model = YOLO(os.path.join(base_dir, SPATULA_MODEL))

    coco_res = coco_model(resized_path, verbose=False)[0]
    spat_res = spatula_model(resized_path, verbose=False)[0]

    person = pick_best_box_by_area(coco_res, target_cls=0, conf_min=0.15)
    if person is None:
        print("❌ COCO did not detect a person.")
        return False

    px1, py1, px2, py2, pconf = person
    bbox = add_margin_bbox((px1, py1, px2, py2), W, H, margin=BBOX_MARGIN)
    bbox = crop_bbox_top(bbox, BBOX_TOP_CROP)
    print(f"✅ COCO PERSON bbox: {bbox}, conf={pconf:.2f}")

    print("Starting OpenPose (BODY only)...")
    params = {
        "model_folder": OPENPOSE_MODELS,
        "model_pose": "BODY_25",
        "face": False,
        "hand": False,
        "render_pose": 0,
        "disable_blending": True,
        "net_resolution": "656x368",
        "num_gpu": 1,
        "num_gpu_start": 0,
        "scale_number": 1,
        "scale_gap": 0.3,
        "render_threshold": 0.05,
    }

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    try:
        datum = op.Datum()
        datum.cvInputData = img
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        if datum.poseKeypoints is None or len(datum.poseKeypoints) == 0:
            print("❌ No pose detected by OpenPose.")
            return False

        kps = datum.poseKeypoints[0]
    finally:
        print("Stopping OpenPose wrapper...")

    out = img.copy()

    # Draw bbox and keypoints
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), 6)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 3)
    draw_keypoints(out, kps, color=(0, 255, 0), conf_thr=KP_CONF_THR)

    # Head
    anchor = get_head_anchor_from_pose(kps, bbox)
    top_head = find_top_of_head_edge(out, bbox, anchor)
    ax, ay = anchor
    hx, hy = top_head
    print(f"✅ Anchor=({ax},{ay}) TopHead=({hx},{hy})")

    draw_dot(out, ax, ay, (255, 0, 0), r=8, text="ANCHOR")
    draw_dot(out, hx, hy, (0, 0, 255), r=10, text="HEAD")

    # Torso: HEAD -> point 8 (Y only)
    torso_y = None
    if MID_HIP < kps.shape[0] and kp_ok(kps[MID_HIP]):
        torso_y = float(kps[MID_HIP][1])
    else:
        ys = []
        if R_HIP < kps.shape[0] and kp_ok(kps[R_HIP]): ys.append(float(kps[R_HIP][1]))
        if L_HIP < kps.shape[0] and kp_ok(kps[L_HIP]): ys.append(float(kps[L_HIP][1]))
        if ys:
            torso_y = float(np.mean(ys))

    if torso_y is None:
        print("❌ Missing pelvis point and hips fallback. Cannot measure.")
        return False

    torso_px = abs(torso_y - float(hy))
    draw_line(out, (hx, hy), (hx, torso_y), color=(0, 255, 255), thick=5)

    # Legs: heel only, compute both, take max
    right_leg_px, right_heel = leg_length_polyline_to_heel(kps, R_HIP, R_KNEE, R_ANKLE, R_HEEL)
    left_leg_px, left_heel = leg_length_polyline_to_heel(kps, L_HIP, L_KNEE, L_ANKLE, L_HEEL)

    candidates = []
    if right_leg_px is not None:
        candidates.append(("RIGHT", right_leg_px, right_heel))
    if left_leg_px is not None:
        candidates.append(("LEFT", left_leg_px, left_heel))

    if not candidates:
        print("❌ No valid legs (missing hip/knee/ankle).")
        return False

    side, leg_px, heel_pt = max(candidates, key=lambda t: t[1])

    if side == "RIGHT":
        hip, knee, ankle = kps[R_HIP], kps[R_KNEE], kps[R_ANKLE]
    else:
        hip, knee, ankle = kps[L_HIP], kps[L_KNEE], kps[L_ANKLE]

    draw_line(out, hip, knee, color=(255, 255, 0), thick=5)
    draw_line(out, knee, ankle, color=(255, 255, 0), thick=5)
    if heel_pt is not None:
        draw_line(out, ankle, heel_pt, color=(255, 255, 0), thick=5)
        draw_dot(out, heel_pt[0], heel_pt[1], (255, 0, 255), r=10, text="HEEL")

    print(f"Torso px (Y-only head->8): {torso_px:.2f}")
    print(f"Leg px (polyline + heel): {leg_px:.2f} ({side})")
    total_px = torso_px + leg_px
    print(f"Total px: {total_px:.2f}")

    # Spatula scale
    spat = pick_best_box_by_area(spat_res, target_cls=SPATULA_CLASS_ID, conf_min=0.15)
    if spat is None:
        print("⚠️ No spatula detected. Cannot convert to cm.")
        length_cm = 0.0
    else:
        sx1, sy1, sx2, sy2, sconf = spat

        # (Good simple scale) use long side of bbox as spatula px length
        spatula_px = max(abs(sx2 - sx1), abs(sy2 - sy1))
        cm_per_px = SPATULA_LENGTH_CM / max(1e-6, spatula_px)
        length_cm = total_px * cm_per_px

        print(f"Spatula conf={sconf:.2f} spatula_px={spatula_px:.2f} cm_per_px={cm_per_px:.6f}")
        print(f"INFANT LENGTH: {length_cm:.2f} cm")

        cv2.rectangle(out, (sx1, sy1), (sx2, sy2), (0, 0, 0), 6)
        cv2.rectangle(out, (sx1, sy1), (sx2, sy2), (0, 255, 255), 3)
        cv2.putText(out, "SPATULA", (sx1, max(18, sy1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(out, f"Length: {length_cm:.1f} cm", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 7)
    cv2.putText(out, f"Length: {length_cm:.1f} cm", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    out_path = os.path.join(output_dir, "result_infant_measurement.png")
    cv2.imwrite(out_path, out)
    print(f"Output saved to: {out_path}")

    opWrapper.stop()
    print("GPU memory freed")
    return length_cm
