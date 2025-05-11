
import cv2
import numpy as np
import csv

IMAGE_PATH = "new_infant.jpeg"
RULER_HEIGHT_CM = 30
OUTPUT_CSV = "measurements_log.csv"

measurements = []

def measure_ruler_pixels(image):
    clone = image.copy()
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            if len(points) == 2:
                cv2.line(clone, points[0], points[1], (255, 0, 0), 2)
                print(f"[INFO] Measured ruler height: {abs(points[1][1] - points[0][1])} pixels")
            cv2.imshow("Measure Ruler", clone)

    cv2.imshow("Measure Ruler", clone)
    cv2.setMouseCallback("Measure Ruler", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return abs(points[1][1] - points[0][1]) if len(points) == 2 else None

def draw_dropdown_and_button(img, selected_index):
    options = ['Head', 'Torso', 'Arm', 'Leg', 'Custom']
    start_x, start_y = 10, 10
    box_width, box_height = 100, 30
    for i, option in enumerate(options):
        top_left = (start_x, start_y + i * (box_height + 5))
        bottom_right = (top_left[0] + box_width, top_left[1] + box_height)
        color = (0, 255, 0) if i == selected_index else (200, 200, 200)
        cv2.rectangle(img, top_left, bottom_right, color, -1)
        cv2.putText(img, option, (top_left[0] + 5, top_left[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Done button
    cv2.rectangle(img, (10, 190), (110, 220), (0, 0, 255), -1)
    cv2.putText(img, "Done", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Undo button
    cv2.rectangle(img, (10, 225), (110, 255), (50, 50, 50), -1)
    cv2.putText(img, "Undo", (30, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def get_selected_option(y_click):
    index = (y_click - 10) // 35
    return index if 0 <= index <= 4 else None

def measure_limb_path(image, pixels_per_cm):
    clone = image.copy()
    points = []
    labels = ['Head', 'Torso', 'Arm', 'Leg', 'Custom']
    selected_index = 0

    def click_limb(event, x, y, flags, param):
        nonlocal selected_index
        if event == cv2.EVENT_LBUTTONDOWN:
            if 10 <= x <= 110 and 10 <= y <= 160:
                sel = get_selected_option(y)
                if sel is not None:
                    selected_index = sel
                return
            elif 10 <= x <= 110 and 190 <= y <= 220:  # Done button
                if len(points) >= 2:
                    total_px = sum(np.hypot(points[i+1][0] - points[i][0], points[i+1][1] - points[i][1]) for i in range(len(points) - 1))
                    total_cm = total_px / pixels_per_cm
                    part = labels[selected_index]
                    label = f"{part}: {total_cm:.2f} cm"
                    print(f"[INFO] {label}")
                    for i in range(len(points) - 1):
                        cv2.line(clone, points[i], points[i+1], (0, 255, 255), 2)
                    cv2.putText(clone, label, (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    measurements.append({"part": part, "px": round(total_px, 2), "cm": round(total_cm, 2)})
                    points.clear()
                return
            elif 10 <= x <= 110 and 225 <= y <= 255:  # Undo button
                if points:
                    points.pop()
                return
            else:
                points.append((x, y))

    cv2.namedWindow("Measure Limb")
    cv2.setMouseCallback("Measure Limb", click_limb)

    while True:
        display = clone.copy()
        for i in range(len(points)):
            cv2.circle(display, points[i], 4, (0, 255, 0), -1)
        for i in range(len(points) - 1):
            cv2.line(display, points[i], points[i+1], (0, 255, 255), 2)
        draw_dropdown_and_button(display, selected_index)
        cv2.imshow("Measure Limb", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cv2.imwrite("annotated_image.jpg", clone)
    cv2.destroyAllWindows()

def export_results_to_csv():
    if not measurements:
        return
    with open(OUTPUT_CSV, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["part", "px", "cm"])
        writer.writeheader()
        writer.writerows(measurements)
    print(f"[INFO] Measurements saved to {OUTPUT_CSV}")

def main():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise ValueError("Image not found.")

    ruler_px_height = measure_ruler_pixels(image)
    if not ruler_px_height:
        return

    pixels_per_cm = ruler_px_height / RULER_HEIGHT_CM
    print(f"[INFO] Calibration = {pixels_per_cm:.2f} px/cm")

    measure_limb_path(image.copy(), pixels_per_cm)
    export_results_to_csv()

if __name__ == "__main__":
    main()
