# 🚗 Lane Detection from Video using OpenCV and Python

This project detects lane lines in road videos using classical computer vision techniques with OpenCV and NumPy. It identifies lane lines, calculates their angles, merges close lines, and overlays the lane markings with angle annotations on the video frames.

---

## 🛠️ Features

- Converts video frames to grayscale, applies Gaussian blur, and detects edges using Canny edge detector.
- Defines a region of interest focusing on the road area to reduce noise.
- Detects lane lines using Hough Line Transform (`cv2.HoughLinesP`).
- Classifies detected lines into left and right lanes based on slope.
- Merges close lane lines to reduce clutter.
- Calculates angles of lane lines and displays them on the video.
- Displays lane lines with color-coded overlays (green for left lane, red for right lane).
- Processes and displays the video in real-time with lane detection overlay.

---

## 💻 Usage

1. Update the `input_video_path` variable in the script with your own video file path.

2. Run the script:

```bash
python your_script_name.py
  ```
The video window will open showing detected lanes and their angles overlaid.
Press q to quit the video window anytime.

⚙️ Requirements
Python 3.x
OpenCV (opencv-python)
NumPy

Install dependencies with:
pip install opencv-python numpy

🔧 How It Works
Edge Detection
The script converts frames to grayscale, applies Gaussian blur, then detects edges using the Canny algorithm.

Region of Interest
Focuses on a polygonal section where lanes are expected to appear, filtering out irrelevant areas.

Lane Line Detection
Uses probabilistic Hough transform to detect line segments in the region of interest.

Line Classification
Separates detected lines into left and right lanes by calculating their slopes.

Line Merging
Combines close lines on each side to form cleaner lane lines.

Angle Calculation
Calculates the angle of each lane line and displays the values near the lines on the video frame.

🚩 Notes
The region of interest polygon may need adjustment depending on your camera angle and video perspective.

This is a basic lane detection approach and may not work perfectly in complex scenarios or with noisy videos.

Works best with clearly visible lane markings and consistent lighting.

yaml
Copy
Edit

