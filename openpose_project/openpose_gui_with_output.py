"""
Infant Length Measurement - GUI Application

This module provides a graphical user interface for the premature infant crown-to-heel
measurement system. It allows users to either browse for images or capture photos using
a live camera feed for analysis.

Features:
    - Browse and select images from file system
    - Live camera capture with camera selection dropdown
    - Real-time processing with OpenPose and YOLO
    - Visual display of annotated measurement results
    - Automatic saving of captured images to 'input' folder

Usage:
    Run directly: python openpose_gui_with_output.py
    
Requirements:
    - tkinter: GUI framework
    - OpenCV: Camera capture and image processing
    - PIL/Pillow: Image display in GUI
    - measured_full_body: Core measurement module

Author: OpenPose Project
Date: 2026
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
import os
from datetime import datetime
from PIL import Image, ImageTk

# Import the core measurement processing module
print("🔄 Loading OpenPose module...")
import measured_full_body  # Core OpenPose processing module

print("✅ OpenPose module loaded!")

# Global variable to store currently selected image path
selected_image_path = None


def process_with_openpose(image_path, label_output):
    """
    Process an image with OpenPose and display the measurement result.
    
    This function serves as the bridge between the GUI and the core measurement
    module. It updates the GUI label with processing status and results, and
    displays the annotated output image in a new window when processing completes.
    
    Args:
        image_path (str): Absolute path to the image file to process
        label_output (tk.Label): Tkinter label widget to update with status/results
        
    Returns:
        None
        
    Side Effects:
        - Updates label_output text and color based on processing status
        - Opens a new window to display the result image if successful
        - Prints error details to console if processing fails
        
    Notes:
        - Shows blue text while processing
        - Shows red text if measurement fails
        - Shows green bold text with measurement result if successful
        - Expects output image at 'output/result_infant_measurement.png'
    """
    try:
        label_output.config(text=f"⏳ Analyzing infant pose...", fg="blue")
        result = measured_full_body.cal_height(image_path)

        if result is False or result == 0:
            label_output.config(text="❌ Measurement failed. Ensure 15cm reference is visible.", fg="red")
        else:
            label_output.config(text=f"✅ Infant Length: {result:.1f} cm", fg="green", font=("Arial", 16, "bold"))

            # Show output image if it exists
            output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output",
                                       "result_infant_measurement.png")
            if os.path.exists(output_path):
                show_result_window(output_path)
    except Exception as e:
        label_output.config(text=f"⚠️ Error: {str(e)}", fg="red")
        print(f"Error details: {e}")


def show_result_window(image_path):
    """
    Display the processed result image in a new popup window.
    
    Creates a Toplevel window that displays the annotated measurement result image.
    The image is automatically resized if it exceeds the maximum display size while
    maintaining aspect ratio.
    
    Args:
        image_path (str): Path to the result image file to display
        
    Returns:
        None
        
    Window Contents:
        - Resized image (max 800x800 pixels) with preserved aspect ratio
        - Close button to dismiss the window
        
    Notes:
        - Uses high-quality LANCZOS resampling for image resizing
        - Keeps image reference to prevent garbage collection
        - Image is converted from BGR (OpenCV) to RGB (Tkinter) format
    """
    result_win = tk.Toplevel()
    result_win.title("📊 Processing Result")

    # Load and display image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img_pil = Image.fromarray(img_rgb)

    # Resize if image is too large for comfortable viewing
    max_size = 800
    if img_pil.width > max_size or img_pil.height > max_size:
        ratio = min(max_size / img_pil.width, max_size / img_pil.height)
        new_size = (int(img_pil.width * ratio), int(img_pil.height * ratio))
        img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)

    imgtk = ImageTk.PhotoImage(image=img_pil)

    # Display image in label (must keep reference to prevent garbage collection)
    lbl_img = tk.Label(result_win, image=imgtk)
    lbl_img.image = imgtk  # Keep a reference
    lbl_img.pack(padx=10, pady=10)

    btn_close = tk.Button(result_win, text="Close", command=result_win.destroy, font=("Arial", 12))
    btn_close.pack(pady=5)


def browse_image(label_output):
    """
    Open file dialog to browse for an image and process it.
    
    Opens a standard file selection dialog allowing the user to choose an image
    file. Once selected, the image is immediately processed through OpenPose.
    
    Args:
        label_output (tk.Label): Label widget to update with processing status
        
    Returns:
        None
        
    Supported Formats:
        - JPEG (.jpg, .jpeg)
        - PNG (.png)
        - Bitmap (.bmp)
        
    Notes:
        - Does nothing if user cancels the file dialog
        - Immediately starts processing after file selection
    """
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path:
        process_with_openpose(file_path, label_output)


def open_camera_window(label_output):
    """
    Open a live camera window for capturing infant photos.
    
    Creates a popup window with live camera feed preview and camera selection options.
    Users can select from available cameras (indices 0-4) and capture photos for
    immediate processing.
    
    Args:
        label_output (tk.Label): Label widget to update with processing results
        
    Returns:
        None
        
    Window Features:
        - Live camera preview at 640x640 pixels
        - Dropdown menu to select camera device (0-4)
        - Capture button to take photo and process
        - Auto-save captured images to 'input' folder with timestamp
        
    Notes:
        - Captured images named as 'captured_YYYYMMDD_HHMMSS.jpg'
        - Camera is released when window closes or photo is captured
        - Shows error message if camera cannot be opened or capture fails
    """
    cam_win = tk.Toplevel()
    cam_win.title("📸 Live Camera")
    cam_win.geometry("640x640")

    lbl_camera = tk.Label(cam_win)
    lbl_camera.pack()

    # Camera selection controls
    camera_label = tk.Label(cam_win, text="Select Camera:", font=("Arial", 12))
    camera_label.pack(pady=5)

    # Check camera indices 0-4 for available cameras
    camera_options = list(range(5))  # Check camera indices 0–4
    camera_var = tk.IntVar(value=0)
    camera_dropdown = ttk.Combobox(cam_win, textvariable=camera_var, values=camera_options, state="readonly")
    camera_dropdown.pack(pady=5)

    # Initialize video capture with default camera
    cap = cv2.VideoCapture(camera_var.get())

    def update_frame():
        """
        Update camera frame in the preview window.
        
        This internal function runs continuously to display the live camera feed.
        It reads frames from the camera, converts color format, and updates the
        display label. Schedules itself to run again after 10ms for smooth video.
        
        Returns:
            None
        """
        nonlocal cap
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB for Tkinter display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            lbl_camera.imgtk = imgtk
            lbl_camera.configure(image=imgtk)
        lbl_camera.after(10, update_frame)  # Update every 10ms for smooth video

    def capture_photo():
        """
        Capture photo from camera and process it.
        
        Captures the current camera frame, saves it to the 'input' folder with
        a timestamp, closes the camera window, and starts OpenPose processing.
        
        Returns:
            None
        """
        ret, frame = cap.read()
        if ret:
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = "input"
            os.makedirs(save_dir, exist_ok=True)
            filename = f"captured_{timestamp}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, frame)
            
            # Clean up camera resources
            cap.release()
            cam_win.destroy()
            
            # Process the captured image
            process_with_openpose(filepath, label_output)
        else:
            messagebox.showerror("Error", "Failed to capture photo")

    def change_camera(*args):
        """
        Switch to a different camera device.
        
        Releases the current camera and opens a new one based on the selected
        index from the dropdown menu.
        
        Args:
            *args: Variable arguments from the trace callback (unused)
            
        Returns:
            None
        """
        nonlocal cap
        cap.release()
        new_index = camera_var.get()
        cap = cv2.VideoCapture(new_index)
        if not cap.isOpened():
            messagebox.showwarning("Warning", f"Could not open camera {new_index}")

    # Bind camera change event to dropdown
    camera_var.trace("w", change_camera)

    btn_capture = tk.Button(cam_win, text="📸 Take Photo", command=capture_photo,
                            font=("Arial", 14), bg="#4CAF50", fg="white", padx=20, pady=10)
    btn_capture.pack(pady=10)

    # Start the camera preview loop
    update_frame()
    
    # Ensure camera is released when window is closed
    cam_win.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), cam_win.destroy()))


# ============================================================================
# Main GUI Setup
# ============================================================================
print("🎨 Creating GUI window...")
root = tk.Tk()
root.title("OpenPose Height Estimator")
root.geometry("500x400")
root.configure(bg="#f5f5f5")
print("✅ GUI window created!")

# ====================
# GUI Layout Elements
# ====================

# Title section
title = tk.Label(root, text="👶 Infant Length Measurement",
                 font=("Helvetica", 20, "bold"), bg="#f5f5f5", fg="#333")
title.pack(pady=20)

# Subtitle section
subtitle = tk.Label(root, text="Premature Infant Crown-to-Heel Measurement System",
                    font=("Arial", 11), bg="#f5f5f5", fg="#666")
subtitle.pack(pady=5)

# Status/output display label
output_label = tk.Label(root, text="No image processed yet",
                        font=("Arial", 13), bg="#f5f5f5", fg="#555")
output_label.pack(pady=20)

# Container frame for buttons
btn_frame = tk.Frame(root, bg="#f5f5f5")
btn_frame.pack(pady=10)

# Browse image file button
btn_browse = tk.Button(btn_frame, text="📁 Browse Image",
                       command=lambda: browse_image(output_label),
                       width=20, font=("Arial", 12),
                       bg="#2196F3", fg="white", padx=10, pady=8,
                       cursor="hand2")
btn_browse.pack(pady=10)

# Open camera button
btn_camera = tk.Button(btn_frame, text="📸 Open Camera",
                       command=lambda: open_camera_window(output_label),
                       width=20, font=("Arial", 12),
                       bg="#FF9800", fg="white", padx=10, pady=8,
                       cursor="hand2")
btn_camera.pack(pady=10)

# Instructions text
instructions = tk.Label(root,
                        text="📋 Instructions:\n• Place infant in supine position (lying flat)\n• Ensure 15cm reference object is visible and parallel to infant\n• Camera overhead, 50-100cm distance\n• Good lighting, contrasting background",
                        font=("Arial", 9), bg="#f5f5f5", fg="#888", justify="left")
instructions.pack(pady=20)

# Footer information
footer = tk.Label(root, text="Medical Infant Measurement System • OpenPose + YOLO",
                  font=("Arial", 9), bg="#f5f5f5", fg="#aaa")
footer.pack(side="bottom", pady=10)


# ============================================================================
# Application Cleanup and Startup
# ============================================================================

def on_closing():
    """
    Clean up resources when closing the application.
    
    Forces Python garbage collection to free any remaining GPU memory or
    other resources before destroying the main window.
    
    Returns:
        None
    """
    import gc
    gc.collect()  # Force garbage collection
    root.destroy()


# Register cleanup handler for window close event
root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the Tkinter event loop (blocks until window is closed)
print("🚀 Starting GUI main loop...")
root.mainloop()
print("👋 GUI closed")
