"""
Infant Length Measurement System using OpenPose and YOLO

This module provides functionality for measuring premature infant crown-to-heel length
using computer vision techniques. It combines YOLO object detection with OpenPose pose
estimation to identify infant body keypoints and calculate physical measurements.

Key Features:
    - Detects 15cm reference object (spatula) for scale calibration
    - Uses YOLO for person detection and bounding box extraction
    - Employs OpenPose for detailed body, face, and hand keypoint detection
    - Calculates vertical distance from head to heel with real-world conversion

Dependencies:
    - OpenCV (cv2): Image processing
    - NumPy: Numerical computations
    - Ultralytics YOLO: Object detection
    - PyOpenPose: Pose estimation
"""

import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO

# Add OpenPose Python bindings to system path
sys.path.append(r'C:\openpose_build\openpose\build\python\openpose\Release')
import pyopenpose as op

# Global variable to store calculated infant height in centimeters
height_cm = 0


def get_spatula_Height(image_path=''):
    """
    Detect and locate the 15cm reference object (spatula) in the image.
    
    This function uses a custom-trained YOLO model to detect a spatula or reference
    object of known length (15cm) in the image. The detected object's bounding box
    coordinates are used for scale calibration.
    
    Args:
        image_path (str): Path to the image file to analyze
        
    Returns:
        tuple: A tuple containing:
            - top_point (tuple): (x, y) coordinates of the top-left corner, or None if not found
            - bottom_point (tuple): (x, y) coordinates of the bottom-right corner, or None if not found
            
    Notes:
        - The function expects a trained YOLO model file 'best.pt' in the current directory
        - Class ID 1 is specifically associated with the 15cm reference object
        - Prints detection details including confidence scores and coordinates
    """
    try:
        spatula_model_path = "best.pt"

        if not os.path.exists(spatula_model_path):
            print(f"Spatula model not found: {spatula_model_path}")
            return None, None

        print(f"Loading spatula detection model: {spatula_model_path}")
        spatula_model = YOLO(spatula_model_path)

        if hasattr(spatula_model, 'names'):
            print(f"Model classes: {spatula_model.names}")

        print(f"Detecting 15cm reference in: {os.path.basename(image_path)}")
        results = spatula_model(image_path, verbose=False)[0]

        total_detections = len(results.boxes)
        print(f"Found {total_detections} object(s)")

        if total_detections > 0:
            for i, box in enumerate(results.boxes.data):
                cls_id = int(box[5])
                conf = float(box[4])
                class_name = spatula_model.names.get(cls_id, f"class_{cls_id}") if hasattr(spatula_model,
                                                                                           'names') else f"class_{cls_id}"
                print(f"Object {i}: {class_name} (class {cls_id}), confidence={conf:.2f}")

        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            cls_id = int(cls)

            if cls_id == 1:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                top_x, top_y = x1, y1
                bottom_x, bottom_y = x2, y2
                print(f"15cm Reference detected!")
                print(f"Point 1: (x={top_x}, y={top_y})")
                print(f"Point 2: (x={bottom_x}, y={bottom_y})")
                print(f"Confidence: {conf:.2f}")
                return (top_x, top_y), (bottom_x, bottom_y)

        print("No spatula detected (class 1 not found)")
        return None, None

    except Exception as e:
        print(f"Error detecting spatula: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def draw_keypoints(image, keypoints, color=(0, 255, 0), radius=5, label=False):
    """
    Draw detected keypoints on an image with optional labels.
    
    This function visualizes pose keypoints by drawing colored circles at each keypoint
    location. It can be used for body joints, facial landmarks, or hand keypoints.
    
    Args:
        image (np.ndarray): The image array to draw on (modified in-place)
        keypoints (np.ndarray): Array of keypoints with shape (N, 3) where each row
                                contains [x, y, confidence]
        color (tuple): BGR color tuple for the keypoint circles (default: green)
        radius (int): Radius of the keypoint circles in pixels (default: 5)
        label (bool): If True, draws keypoint index numbers next to each point (default: False)
        
    Returns:
        None: Modifies the input image in-place
        
    Notes:
        - Only draws keypoints with confidence > 0.05
        - Adds a black outline around each circle for better visibility
        - Labels are drawn in white with colored outline when enabled
    """
    if keypoints is None or len(keypoints) == 0:
        return
    for i in range(keypoints.shape[0]):
        keypoint = keypoints[i].flatten()
        if keypoint.shape[0] >= 3:
            x, y, confidence = float(keypoint[0]), float(keypoint[1]), float(keypoint[2])
            if confidence > 0.05:
                cv2.circle(image, (int(x), int(y)), radius + 2, (0, 0, 0), -1)
                cv2.circle(image, (int(x), int(y)), radius, color, -1)
                if label:
                    cv2.putText(image, str(i), (int(x) + 8, int(y) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                    cv2.putText(image, str(i), (int(x) + 8, int(y) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def distance(p1, p2):
    """
    Calculate Euclidean distance between two 2D points.
    
    Args:
        p1 (tuple or list): First point coordinates (x1, y1)
        p2 (tuple or list): Second point coordinates (x2, y2)
        
    Returns:
        float: Euclidean distance between the two points in pixels
        
    Example:
        >>> distance((0, 0), (3, 4))
        5.0
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))


def cal_height(image_path=""):
    """
    Calculate infant crown-to-heel length from an image using pose estimation.
    
    This is the main processing function that orchestrates the entire measurement pipeline:
    1. Resizes the input image to 640x480 for consistent processing
    2. Detects the infant using YOLO person detection
    3. Runs OpenPose to extract detailed body, face, and hand keypoints
    4. Identifies the highest point (head) and lowest point (heel/ankle)
    5. Calculates vertical distance in pixels
    6. Converts to centimeters using the 15cm reference object
    7. Generates an annotated output image with measurements
    
    Args:
        image_path (str): Absolute path to the input image file
        
    Returns:
        float or bool: 
            - Measured infant length in centimeters if successful
            - 0 if reference object not found (returns pixel measurement only)
            - False if measurement failed
            
    Global Variables:
        height_cm (float): Updated with the calculated height
        
    Output Files:
        - Creates 'output/result_infant_measurement.png' with annotated results
        - Creates resized version of input image with '_resized' suffix
        
    Notes:
        - Requires 15cm reference object in image for accurate measurements
        - Image is resized to 640x480 for consistent processing
        - Uses BODY_25 model for pose estimation with face and hand detection
        - Assumes infant is in supine position (lying flat)
    """
    global height_cm

    # Setup directory paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, "output")
    model_dir = r"C:\openpose_build\openpose\models"
    yolo_model_path = os.path.join(base_dir, "yolov8n.pt")

    # Load and validate input image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return False

    original_h, original_w = img.shape[:2]
    print(f"Original image size: {original_w}x{original_h}")

    # Resize image to standard dimensions for consistent processing
    target_w, target_h = 640, 480
    resized_img = cv2.resize(img, (target_w, target_h))

    base_name = os.path.basename(image_path)
    name_parts = os.path.splitext(base_name)
    resized_name = name_parts[0] + "_resized" + name_parts[1]
    resized_path = os.path.join(os.path.dirname(image_path), resized_name)

    cv2.imwrite(resized_path, resized_img)
    print(f"Resized image saved to: {resized_path}")
    print(f"New size: {target_w}x{target_h}")

    # Use resized image for all subsequent processing
    image_path = resized_path

    # Step 1: Person Detection using YOLO
    try:
        print("Loading YOLO model for person detection...")
        if not os.path.exists(yolo_model_path):
            print(f"Downloading yolov8n.pt to {yolo_model_path}...")
            yolo_person_model = YOLO("yolov8n.pt")
            import shutil
            default_location = os.path.join(os.path.expanduser("~"), ".cache", "ultralytics", "yolov8n.pt")
            if os.path.exists(default_location):
                shutil.copy(default_location, yolo_model_path)
                print(f"Model saved to project folder")
        else:
            print(f"Using existing yolov8n.pt: {yolo_model_path}")
            yolo_person_model = YOLO(yolo_model_path)

        print("Running YOLO person detection at 640x480...")
        yolo_results = yolo_person_model(image_path, imgsz=640, verbose=False)[0]
    except Exception as e:
        print(f"YOLO error: {e}")
        return False

    # Extract person bounding box from YOLO detections
    person_box = None
    for box in yolo_results.boxes.data:
        cls = int(box[5])
        if cls == 0:
            person_box = box[:4].cpu().numpy().astype(int)
            conf = float(box[4])
            print(f"Person detected with confidence: {conf:.2f}")
            break

    if person_box is None:
        print("No person detected by YOLO.")
        return False

    # Calculate head position from bounding box top-center
    x1, y1, x2, y2 = person_box
    top_x = int((x1 + x2) / 2)  # Center X coordinate
    top_y = int(y1)  # Top Y coordinate
    print(f"Infant bounding box: x={x1}-{x2}, y={y1}-{y2}")

    # Step 2: Configure OpenPose parameters for full-body analysis
    params = {
        "model_folder": model_dir,
        "face": True,
        "hand": True,
        "model_pose": "BODY_25",
        "num_gpu": 1,
        "num_gpu_start": 0,
        "net_resolution": "640x368",
        "scale_number": 2,
        "scale_gap": 0.25,
        "face_net_resolution": "368x368",
        "hand_net_resolution": "368x368",
        "render_threshold": 0.05,
        "alpha_pose": 0.6,
        "render_pose": 2,
        "disable_blending": False,
    }

    print("Starting OpenPose with FULL features (body + face + hands)...")

    # Initialize and configure OpenPose wrapper
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Step 3: Run pose estimation
    try:
        image = cv2.imread(image_path)
        datum = op.Datum()
        datum.cvInputData = image
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))  # Process image
        output_img = datum.cvOutputData

        if datum.poseKeypoints is None or len(datum.poseKeypoints) == 0:
            print("No pose detected by OpenPose.")
            opWrapper.stop()
            return False

        print(f"Detected {len(datum.poseKeypoints)} person(s)")

        if datum.faceKeypoints is not None and len(datum.faceKeypoints) > 0:
            print(f"Face detected: {datum.faceKeypoints.shape}")
        else:
            print(f"No face detected")

        if datum.handKeypoints is not None and len(datum.handKeypoints) > 0:
            print(f"Hands detected")
        else:
            print(f"No hands detected")

    except Exception as e:
        print(f"OpenPose error: {e}")
        opWrapper.stop()
        return False
    finally:
        print("Stopping OpenPose wrapper...")

    # Step 4: Visualize detected keypoints on output image
    # Draw body keypoints in green with labels
    draw_keypoints(output_img, datum.poseKeypoints[0], (0, 255, 0), radius=5, label=True)

    if datum.faceKeypoints is not None and len(datum.faceKeypoints) > 0:
        draw_keypoints(output_img, datum.faceKeypoints[0], (255, 255, 0), radius=2, label=False)
        print("Face keypoints drawn (yellow)")

    if datum.handKeypoints is not None and len(datum.handKeypoints) > 0:
        try:
            if datum.handKeypoints[0] is not None and len(datum.handKeypoints[0]) > 0:
                draw_keypoints(output_img, datum.handKeypoints[0][0], (255, 0, 255), radius=3, label=False)
                print("Left hand keypoints drawn (magenta)")
            if datum.handKeypoints[0] is not None and len(datum.handKeypoints[0]) > 1:
                draw_keypoints(output_img, datum.handKeypoints[0][1], (0, 255, 255), radius=3, label=False)
                print("Right hand keypoints drawn (cyan)")
        except:
            print("Could not draw hand keypoints")

    # Mark the detected head position from YOLO bounding box
    cv2.circle(output_img, (top_x, top_y), 8, (0, 0, 255), -1)
    cv2.circle(output_img, (top_x, top_y), 10, (255, 255, 255), 2)
    cv2.putText(output_img, "Head", (top_x + 12, top_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
    cv2.putText(output_img, "Head", (top_x + 12, top_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Step 5: Calculate infant length measurement
    try:
        # Extract relevant body keypoints from BODY_25 model
        nose = datum.poseKeypoints[0][0]
        neck = datum.poseKeypoints[0][1]
        r_ankle = datum.poseKeypoints[0][11]
        l_ankle = datum.poseKeypoints[0][14]
        r_heel = datum.poseKeypoints[0][24]
        l_heel = datum.poseKeypoints[0][21]

        # Adjust head position for better alignment with actual head top
        # The YOLO bounding box tends to be slightly off-center
        head_x = top_x - 50  # Shift left to better center on head
        head_y = top_y + 20  # Shift down to approximate crown position

        print(f"Original head position: ({top_x}, {top_y})")
        print(f"Adjusted head position: ({head_x}, {head_y})")

        # Find the lowest detected point among all foot/ankle keypoints
        lowest_point = None
        lowest_y = 0

        # Check all foot/ankle points and find the lowest
        foot_points = [
            ("right heel", r_heel),
            ("left heel", l_heel),
            ("right ankle", r_ankle),
            ("left ankle", l_ankle)
        ]

        for name, point in foot_points:
            if point[2] > 0.05:  # confidence check
                if point[1] > lowest_y:  # Y coordinate (lower = higher value in image)
                    lowest_y = point[1]
                    lowest_point = point
                    lowest_name = name

        if lowest_point is None:
            print("Could not detect feet/ankles")
            opWrapper.stop()
            return False

        # Define measurement points: vertical line from head to heel projection
        top_point = (head_x, head_y)
        bottom_point = (head_x, int(lowest_point[1]))

        print(f"Top point (head): {top_point}")
        print(f"Bottom point ({lowest_name}): ({int(lowest_point[0])}, {int(lowest_point[1])})")
        print(f"Projected bottom: {bottom_point}")

        # Visualize the measurement line (vertical distance)
        cv2.line(output_img, top_point, bottom_point, (0, 0, 0), thickness=5)  # Black outline
        cv2.line(output_img, top_point, bottom_point, (255, 0, 0), thickness=3)  # Blue line

        # Draw projection guide showing how heel maps to vertical measurement line
        num_dots = 10
        for i in range(num_dots):
            t = i / (num_dots - 1)
            x = int(lowest_point[0] + t * (bottom_point[0] - lowest_point[0]))
            y = int(lowest_point[1] + t * (bottom_point[1] - lowest_point[1]))
            cv2.circle(output_img, (x, y), 2, (0, 255, 255), -1)

        # Calculate pixel distance (straight vertical measurement)
        total_px = abs(bottom_point[1] - top_point[1])

        print(f"Vertical distance: {total_px:.2f} px")
        print("Searching for 15cm reference object...")

        # Step 6: Convert pixels to centimeters using reference object
        pt1, pt2 = get_spatula_Height(image_path=resized_path)

        if pt1 is None or pt2 is None:
            print("WARNING: No 15cm reference found!")
            print("Cannot calculate length in cm - pixel measurement only")
            height_cm = 0
        else:
            # Calculate scale factor from reference object
            pixel_distance = distance(pt1, pt2)
            known_length_cm = 15.0  # Known physical length of reference
            scale_cm_per_px = known_length_cm / pixel_distance
            height_cm = total_px * scale_cm_per_px  # Convert measurement to cm

            print(f"Reference length: {pixel_distance:.2f} pixels = 15.0 cm")
            print(f"Scale factor: {scale_cm_per_px:.4f} cm/pixel")
            print(f"INFANT LENGTH: {height_cm:.2f} cm")

            # Visualize the reference object on output image
            cv2.line(output_img, pt1, pt2, (0, 255, 255), thickness=4)
            cv2.putText(output_img, "15cm REF", (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Display final measurement on image
            cv2.putText(output_img, f"Length: {height_cm:.1f} cm",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5)
            cv2.putText(output_img, f"Length: {height_cm:.1f} cm",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Also display pixel measurement for reference
        cv2.putText(output_img, f"Pixels: {int(total_px)} px",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    except Exception as e:
        print(f"Error calculating length: {e}")
        import traceback
        traceback.print_exc()
        opWrapper.stop()
        return False

    # Step 7: Save annotated output image
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    out_file = os.path.join(output_path, "result_infant_measurement.png")
    cv2.imwrite(out_file, output_img)
    print(f"Output saved to: {out_file}")

    # Clean up OpenPose resources
    opWrapper.stop()
    print("GPU memory freed")

    return height_cm


# ============================================================================
# Main Execution Block
# ============================================================================
if __name__ == "__main__":
    """
    Command-line interface for testing the measurement system.
    
    Usage:
        python measured_full_body.py <image_path>
        
    Example:
        python measured_full_body.py input/infant_photo.jpg
    """
    import sys

    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        if os.path.exists(test_image):
            result = cal_height(test_image)
            if result:
                print(f"\n{'='*50}")
                print(f"FINAL MEASUREMENT: {result:.2f} cm")
                print(f"{'='*50}")
        else:
            print(f"Error: Image not found: {test_image}")
            print(f"Usage: python measured_full_body.py <image_path>")
    else:
        print("Usage: python measured_full_body.py <image_path>")
        print("Example: python measured_full_body.py input/infant_photo.jpg")
