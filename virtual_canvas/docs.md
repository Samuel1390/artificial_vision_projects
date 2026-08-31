# Virtual Canvas Setup & Prerequisites Guide

This guide details the step-by-step setup required to run the Virtual Canvas gesture tracking application on Windows, macOS, and Linux.

---

## 1. Download the Hand Landmarker Model

MediaPipe requires a pre-trained machine learning model file to process hand tracking landmarks.

- **Download Link:** [MediaPipe Hand Landmarker Task Model (Bundle)](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)
- **File Placement:** Download the file and place it exactly inside a folder named `models` within your parent directory:
  ```text
  your_project_folder/
  ├── models/
  │   └── hand_landmarker.task
  └── src/
      ├── main.py
      ├── draw_hand.py
      └── is_hand_closed.py
  ```

---

## 2. Operating System Prerequisites

Before running `pip install`, specific system-level dependencies must be present.

### Windows

1. Ensure you have **Python 3.9 - 3.11** installed and added to your system `PATH`.
2. Install the C++ build features (required for underlying native extensions):
   - Download the [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
   - Select **Desktop development with C++** during installation.

### macOS

1. Open your terminal and install the command-line developer suite if you haven't already:
   ```bash
   xcode-select --install
   ```

made with ❤️ by Samuel nelo / instagram @samuel_nelo.py
