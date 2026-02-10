import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
import os
from datetime import datetime
from PIL import Image, ImageTk

# Import measured_full_body at the top
print("🔄 Loading OpenPose module...")
import measured_full_body  # Your OpenPose processing module

print("✅ OpenPose module loaded!")

selected_image_path = None


def process_with_openpose(image_path, label_output):
    """Process image with OpenPose and display result"""
    try:
        label_output.config(text=f"⏳ Analyzing infant pose...", fg="blue")
        result = measured_full_body.cal_height(image_path)

        if result is False or result == 0:
            label_output.config(text="❌ Measurement failed.\n Ensure 15cm reference object is visible.", fg="red")
        else:
            label_output.config(text=f"✅ Infant Length: {result:.1f} cm", fg="green", font=("Arial", 16, "bold"))

            # # Show output image if it exists
            # output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output",
            #                            "result_infant_measurement.png")
            # if os.path.exists(output_path):
            #     show_result_window(output_path)
    except Exception as e:
        label_output.config(text=f"⚠️ Error: {str(e)}", fg="red")
        print(f"Error details: {e}")


def show_result_window(image_path):
    """Display the processed result in a new window"""
    result_win = tk.Toplevel()
    result_win.title("📊 Processing Result")

    # Load and display image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # Resize if too large
    max_size = 800
    if img_pil.width > max_size or img_pil.height > max_size:
        ratio = min(max_size / img_pil.width, max_size / img_pil.height)
        new_size = (int(img_pil.width * ratio), int(img_pil.height * ratio))
        img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)

    imgtk = ImageTk.PhotoImage(image=img_pil)

    lbl_img = tk.Label(result_win, image=imgtk)
    lbl_img.image = imgtk  # Keep a reference
    lbl_img.pack(padx=10, pady=10)

    btn_close = tk.Button(result_win, text="Close", command=result_win.destroy, font=("Arial", 12))
    btn_close.pack(pady=5)


def browse_image(label_output):
    """Browse for an image file and process it"""
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path:
        process_with_openpose(file_path, label_output)


def open_camera_window(label_output):
    """Open camera window for live capture"""
    cam_win = tk.Toplevel()
    cam_win.title("📸 Live Camera")
    cam_win.geometry("640x640")

    lbl_camera = tk.Label(cam_win)
    lbl_camera.pack()

    # Camera selection dropdown
    camera_label = tk.Label(cam_win, text="Select Camera:", font=("Arial", 12))
    camera_label.pack(pady=5)

    camera_options = list(range(5))  # Check camera indices 0–4
    camera_var = tk.IntVar(value=0)
    camera_dropdown = ttk.Combobox(cam_win, textvariable=camera_var, values=camera_options, state="readonly")
    camera_dropdown.pack(pady=5)

    cap = cv2.VideoCapture(camera_var.get(), cv2.CAP_DSHOW)

    def update_frame():
        """Update camera frame"""
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
        """Capture photo from camera"""
        ret, frame = cap.read()
        if ret:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = "input"
            os.makedirs(save_dir, exist_ok=True)
            filename = f"captured_{timestamp}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, frame)
            cap.release()
            cam_win.destroy()
            process_with_openpose(filepath, label_output)
        else:
            messagebox.showerror("Error", "Failed to capture photo")

    def change_camera(*args):
        """Switch camera source"""
        nonlocal cap
        cap.release()
        new_index = camera_var.get()
        cap = cv2.VideoCapture(new_index)
        if not cap.isOpened():
            messagebox.showwarning("Warning", f"Could not open camera {new_index}")

    camera_var.trace("w", change_camera)

    btn_capture = tk.Button(cam_win, text="📸 Take Photo", command=capture_photo,
                            font=("Arial", 14), bg="#4CAF50", fg="white", padx=20, pady=10)
    btn_capture.pack(pady=10)

    update_frame()
    cam_win.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), cam_win.destroy()))


# === Main GUI Setup ===
print("🎨 Creating GUI window...")
root = tk.Tk()
root.title("OpenPose Height Estimator")
root.geometry("500x400")
root.configure(bg="#f5f5f5")
print("✅ GUI window created!")

# Title
title = tk.Label(root, text="👶 Infant Length Measurement",
                 font=("Helvetica", 20, "bold"), bg="#f5f5f5", fg="#333")
title.pack(pady=20)

# Subtitle
subtitle = tk.Label(root, text="Premature Infant Crown-to-Heel Measurement System",
                    font=("Arial", 11), bg="#f5f5f5", fg="#666")
subtitle.pack(pady=5)

# Output label
output_label = tk.Label(root, text="No image processed yet",
                        font=("Arial", 13), bg="#f5f5f5", fg="#555")
output_label.pack(pady=20)

# Buttons frame
btn_frame = tk.Frame(root, bg="#f5f5f5")
btn_frame.pack(pady=10)

# Browse button
btn_browse = tk.Button(btn_frame, text="📁 Browse Image",
                       command=lambda: browse_image(output_label),
                       width=20, font=("Arial", 12),
                       bg="#2196F3", fg="white", padx=10, pady=8,
                       cursor="hand2")
btn_browse.pack(pady=10)

# Camera button
btn_camera = tk.Button(btn_frame, text="📸 Open Camera",
                       command=lambda: open_camera_window(output_label),
                       width=20, font=("Arial", 12),
                       bg="#FF9800", fg="white", padx=10, pady=8,
                       cursor="hand2")
btn_camera.pack(pady=10)

# Instructions
instructions = tk.Label(root,
                        text="📋 Instructions:\n• Place infant in supine position (lying flat)\n• Ensure 15cm reference object is visible and parallel to infant\n• Camera overhead, 50-100cm distance\n• Good lighting, contrasting background",
                        font=("Arial", 9), bg="#f5f5f5", fg="#888", justify="left")
instructions.pack(pady=20)

# Footer
footer = tk.Label(root, text="Medical Infant Measurement System • OpenPose + YOLO",
                  font=("Arial", 9), bg="#f5f5f5", fg="#aaa")
footer.pack(side="bottom", pady=10)


# Cleanup on close
def on_closing():
    """Ensure all resources are freed when closing"""
    import gc
    gc.collect()  # Force garbage collection
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)
print("🚀 Starting GUI main loop...")
root.mainloop()
print("👋 GUI closed")