import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
import os
from datetime import datetime
from PIL import Image, ImageTk
import subprocess
import measured_full_body  # Assuming your main OpenPose code is here and imported

selected_image_path = None

# === Replace this with your OpenPose pipeline ===
def process_with_openpose(image_path, label_output):
    try:
        result = measured_full_body.cal_height(image_path)
        if result is False:
            label_output.config(text="❌ Failed to detect spatula.")
        else:
            label_output.config(text=f"🧍 Estimated Height: {result:.2f} cm")
    except Exception as e:
        label_output.config(text=f"⚠️ Error: {str(e)}")

# === Browse and Auto-Process Image ===
def browse_image(label_output):
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path:
        process_with_openpose(file_path, label_output)

# === Camera Window for Live Photo ===
def open_camera_window(label_output):
    cam_win = tk.Toplevel()
    cam_win.title("📸 Live Camera")
    cam_win.geometry("640x640")

    lbl_camera = tk.Label(cam_win)
    lbl_camera.pack()

    # Dropdown for selecting camera index
    camera_label = tk.Label(cam_win, text="Select Camera:", font=("Arial", 12))
    camera_label.pack(pady=5)

    camera_options = list(range(5))  # Check camera indices 0–4
    camera_var = tk.IntVar(value=0)
    camera_dropdown = ttk.Combobox(cam_win, textvariable=camera_var, values=camera_options, state="readonly")
    camera_dropdown.pack(pady=5)

    cap = cv2.VideoCapture(camera_var.get())

    def update_frame():
        nonlocal cap
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            lbl_camera.imgtk = imgtk
            lbl_camera.configure(image=imgtk)
        lbl_camera.after(10, update_frame)

    def capture_photo():
        ret, frame = cap.read()
        if ret:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = "input"
            os.makedirs(save_dir, exist_ok=True)
            filename = f"captured_{timestamp}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, frame)
            cam_win.destroy()
            cap.release()
            process_with_openpose(filepath, label_output)

    def change_camera(*args):
        nonlocal cap
        cap.release()
        cap = cv2.VideoCapture(camera_var.get())

    camera_var.trace("w", change_camera)

    btn_capture = tk.Button(cam_win, text="📸 Take Photo", command=capture_photo, font=("Arial", 14))
    btn_capture.pack(pady=10)

    update_frame()
    cam_win.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), cam_win.destroy()))

# === Main GUI Setup ===
root = tk.Tk()
root.title("OpenPose Pro GUI")
root.geometry("480x360")
root.configure(bg="#f0f0f0")

title = tk.Label(root, text="🧠 OpenPose Image Processor", font=("Helvetica", 18, "bold"), bg="#f0f0f0")
title.pack(pady=10)

output_label = tk.Label(root, text="No output yet", font=("Arial", 12), bg="#f0f0f0")
output_label.pack(pady=10)

btn_browse = tk.Button(root, text="📁 Browse Image from Computer", command=lambda: browse_image(output_label), width=35, font=("Arial", 12))
btn_camera = tk.Button(root, text="📸 Open Live Camera", command=lambda: open_camera_window(output_label), width=35, font=("Arial", 12))

btn_browse.pack(pady=10)
btn_camera.pack(pady=10)

root.mainloop()
