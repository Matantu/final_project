# Infant Length Measurement System

A standalone **offline** desktop application that estimates **premature infant head-to-heel length** from a **single top‑down RGB image**.

It uses:
- **YOLOv8** (Ultralytics) to detect a **15 cm reference object (spadel)** for pixel→cm calibration
- **OpenPose (pyopenpose)** to extract infant body keypoints
- **Python + Tkinter** for a simple GUI intended for medical staff

---

## 1) System Requirements

### Hardware
- **NVIDIA GPU with CUDA support (minimum 8GB VRAM)** (recommended: RTX 3070 Ti or better)
- RAM: 16GB minimum (32GB recommended)
- SSD: 50GB free

### Software
- Windows 10/11
- NVIDIA driver + **CUDA 11.8** + **cuDNN 8.6** (for CUDA 11.x)
- **Python 3.8.10 (64‑bit)** (recommended for OpenPose Python API compatibility)
- Visual Studio 2019 + CMake (only if you build OpenPose from source)

---

## 2) Quick Start (Project)

If using Git:
```bash
git clone https://github.com/Matantu/final_project.git
cd final_project/openpose_project
```

- `OPENPOSE_PYTHON_PATH` = `C:\openpose_build\openpose\build\python\openpose\Release`


---

## 3) Run

### GUI (recommended)
```bash
python openpose_gui_with_output.py
```

### CLI (batch / single image)
```bash
python measured_full_body.py --input "input\baby_47cm_01.jpg" --output_dir "output"
```

---

## 4) How to Use (Nominal flow)

1. Place the infant (or test doll) in a **top‑down** view.
2. Place the **15 cm spadel** fully visible in the same frame.
3. In the GUI:
   - **Open Camera** → take a photo, or
   - **Browse Image** → select an image file
4. The system:
   - Detects the spadel (YOLOv8) → computes pixel→cm scale
   - Detects keypoints (OpenPose) → chooses head and heel
   - Computes head‑to‑heel distance in cm
5. The GUI shows:
   - The measured length
   - An annotated image for visual verification

---

## 5) Notes

- The system is a **feasibility prototype**  and was validated in non‑clinical conditions using an infant‑sized doll.
- No images are uploaded to cloud services; processing is **local/offline**.

---

## 6) Repository Structure

- `openpose_gui_with_output.py` – Tkinter GUI
- `measured_full_body.py` – core measurement pipeline (YOLO + OpenPose + geometry)
- `input/` – sample test images
- `output/` – generated results (can be empty)

