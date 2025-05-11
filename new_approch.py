import cv2
import numpy as np

IMAGE_PATH = "new_infant.jpeg"
RULER_HEIGHT_CM = 30

def measure_ruler_pixels(image):
    clone = image.copy()
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            if len(points) == 2:
                cv2.line(clone, points[0], points[1], (255, 0, 0), 2)
                height_px = abs(points[1][1] - points[0][1])
                print(f"[INFO] Measured ruler height: {height_px} pixels")
            cv2.imshow("Measure Ruler", clone)

    cv2.imshow("Measure Ruler", clone)
    cv2.setMouseCallback("Measure Ruler", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) == 2:
        return abs(points[1][1] - points[0][1])
    else:
        print("[ERROR] You must click two points to measure the ruler height.")
        return None

def measure_limb_path(image, pixels_per_cm):
    clone = image.copy()
    points = []

    def click_limb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(clone, (x, y), 4, (0, 255, 0), -1)
            if len(points) > 1:
                cv2.line(clone, points[-2], points[-1], (0, 255, 255), 2)
            cv2.imshow("Measure Limb", clone)

    print("[INFO] Click along the limb. Press ESC when done.")
    cv2.imshow("Measure Limb", clone)
    cv2.setMouseCallback("Measure Limb", click_limb)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 and len(points) >= 2:  # ESC key
            break

    cv2.destroyAllWindows()

    total_px = 0
    for i in range(len(points) - 1):
        dx = points[i+1][0] - points[i][0]
        dy = points[i+1][1] - points[i][1]
        segment = np.sqrt(dx**2 + dy**2)
        total_px += segment

    total_cm = total_px / pixels_per_cm
    print(f"[INFO] Total curved length: {total_px:.2f} px = {total_cm:.2f} cm")

    # Annotate on image
    label = f"Length: {total_cm:.2f} cm"
    cv2.putText(clone, label, (points[0][0], points[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow("Final Measurement", clone)
    cv2.imwrite("manual_limb_length_result.jpg", clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise ValueError("Image not found.")

    ruler_px_height = measure_ruler_pixels(image)
    if not ruler_px_height:
        return

    pixels_per_cm = ruler_px_height / RULER_HEIGHT_CM
    print(f"[INFO] Calibration: {pixels_per_cm:.2f} px/cm")

    measure_limb_path(image, pixels_per_cm)

if __name__ == "__main__":
    main()
