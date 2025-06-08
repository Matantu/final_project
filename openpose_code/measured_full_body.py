import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
import sys


# OpenPose setup
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, 'build/python'))
from openpose import pyopenpose as op
def get_spatula_Height(image_path=''):
    model = YOLO("best.pt")  # use your trained model

    # Run inference
    results = model(image_path)[0]

    # Get spatula box (assuming class 0 = spatula)
    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box
        if int(cls) == 1:  # Replace 0 with spatula's class index if different
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Top-left and bottom-right
            top_x, top_y = x1, y1
            bottom_x, bottom_y = x2, y2

            print(f"🧾 Spatula detected:")
            print(f"Top-Left:    (x={top_x}, y={top_y})")
            print(f"Bottom-Right:(x={bottom_x}, y={bottom_y})")
            return (top_x,top_y),(top_x,bottom_y)
        else:
            print("❌ No spatula detected.")
    return None,None

def draw_keypoints(image, keypoints, color=(0, 255, 0), radius=3, label=False):
    if keypoints is None or len(keypoints) == 0:
        return
    for i in range(keypoints.shape[0]):
        keypoint = keypoints[i].flatten()
        if keypoint.shape[0] >= 3:
            x, y, confidence = float(keypoint[0]), float(keypoint[1]), float(keypoint[2])
            if confidence > 0.05:
                cv2.circle(image, (int(x), int(y)), radius, color, -1)
                if label:
                    cv2.putText(image, str(i), (int(x), int(y) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def cal_height(image_path=""):
    global height_cm
    base_dir = os.path.dirname(os.path.abspath(__file__))
    #image_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base_dir, "input", "resized.jpg")
    os.system(f"sudo convert {image_path}  -resize 320x320 {image_path}")
    output_path =  os.path.join(base_dir, "output")
    # output_path = "/home/matan/openpose/output"
    model_dir = os.path.join(base_dir, "models")
    #  model_dir = "/home/matan/openpose/models"

    # Load YOLO
    model = YOLO("yolov8n.pt")
    yolo_results = model(image_path)[0]

    # Detect person bbox
    person_box = None
    for box in yolo_results.boxes.data:
        cls = int(box[5])
        if cls == 0:  # person
            person_box = box[:4].cpu().numpy().astype(int)
            break
    if person_box is None:
        print("❌ No person detected by YOLO.")
        return
    x1, y1, x2, y2 = person_box
    top_x = int((x1 + x2) / 2)
    top_y = int(y1) 

    # Setup OpenPose
    params = {
        "model_folder": model_dir,
        "face": False,  # ❌ disable face detection
        "hand": False,  # ❌ disable hand detection
        "model_pose": "BODY_25"
    }

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Run OpenPose
    image = cv2.imread(image_path)
    datum = op.Datum()
    datum.cvInputData = image
    datum_vector = op.VectorDatum()
    datum_vector.append(datum)
    opWrapper.emplaceAndPop(datum_vector)
    output_img = datum.cvOutputData

    # Draw keypoints
    draw_keypoints(output_img, datum.poseKeypoints[0], (0, 255, 0), label=True)
    #  draw_keypoints(output_img, datum.faceKeypoints[0], (255, 255, 255), label=True)
    #draw_keypoints(output_img, datum.handKeypoints[0], (0, 0, 255), label=True)
    #draw_keypoints(output_img, datum.handKeypoints[1], (255, 0, 0), label=True)

    # Draw top head point (YOLO)
    cv2.circle(output_img, (top_x, top_y), 6, (0, 0, 255), -1)
    cv2.putText(output_img, "Top Head (YOLO)", (top_x + 5, top_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Estimate and draw height lines
    try:
        # face_kpt = datum.faceKeypoints[0][9]   # chin
        torso_kpt = datum.poseKeypoints[0][8]  # mid torso
        hip_right_kpt = datum.poseKeypoints[0][9]  # right torso
        knee_right_kpt = datum.poseKeypoints[0][10]
        ankel_right_kpt = datum.poseKeypoints[0][11]
        foot_kpt = datum.poseKeypoints[0][24]  # left foot
        if foot_kpt[2] < 0.05:
            foot_kpt = datum.poseKeypoints[0][22]  # fallback to right foot

        if torso_kpt[2] > 0.05 and foot_kpt[2] > 0.05:
            face_kpt = [0,0]
            chin = (int(top_x), int(face_kpt[1]))
            torso = (int(top_x), int(torso_kpt[1]))
            between_torso = (int(top_x),int(hip_right_kpt[1]))
            hip_right= (int(hip_right_kpt[0]),int(hip_right_kpt[1]))
            knee_right = (int(knee_right_kpt[0]),int(knee_right_kpt[1]))
            ankel_right = (int(ankel_right_kpt[0]),int(ankel_right_kpt[1]))
            foot = (int(foot_kpt[0]), int(foot_kpt[1]))

            # Draw 3 parts
            # cv2.line(output_img,, chin, (0, 255, 255),thickness=1)
            # cv2.line(output_img, chin, torso, (255, 255, 0),thickness=1)
            cv2.line(output_img,  (top_x, top_y), between_torso, (255, 255, 0),thickness=1)
            cv2.line(output_img, hip_right, knee_right, (255, 0, 255),thickness=1)
            cv2.line(output_img, knee_right, ankel_right, (255, 0, 255),thickness=1)
            # cv2.line(output_img, ankel_right, foot, (255, 0, 255),thickness=5)

            # Calculate total height in pixels
            total_px = distance((top_x,top_y), between_torso) + \
                       distance(hip_right,knee_right) + \
                       distance(knee_right,ankel_right) 
                       # distance(ankel_right, foot)
            print("🧍 Estimated height in pixels:", total_px)
            #pt1 = (396, 280)
            # pt2 = (386, 450)
            pt1,pt2 = get_spatula_Height(image_path=image_path)
            if pt1 is None or pt2 is None:
                return False
            print(pt1,pt2)
            pixel_distance = np.linalg.norm(np.array(pt1) - np.array(pt2))
            known_length_cm = 15 # for example, a 15 cm ruler
            scale_cm_per_px = known_length_cm / pixel_distance
            height_px = total_px
            height_cm = height_px * scale_cm_per_px
            print(f'the total cm height is {height_cm}')
            cv2.putText(output_img, f"Height: {int(total_px)} px",
                        (top_x, foot[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(output_img, f"Height in cm: {int(height_cm)} cm",
                        (top_x, foot[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    except Exception as e:
        print("⚠️ Error estimating height:", e)

    # Save output
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    out_file = os.path.join(output_path, "result_yolo_pose_height_2.png")
    cv2.imwrite(out_file, output_img)
    print("✅ Output saved to:", out_file)
    return height_cm

if __name__ == "__main__":
    main()
