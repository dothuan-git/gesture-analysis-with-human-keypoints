# gesture-analysis-with-human-keypoints

Python tooling for extracting, inspecting, and exporting human gesture keypoints from video.  
The `src/base` pipeline wraps [MediaPipe](https://developers.google.com/mediapipe) detectors to produce synchronized CSVs of face and hand landmarks plus frame-level visualizations.

## Base Pipeline Overview (`src/base`)
- `1.visualization.py` - Runs the **MediaPipe Solutions** (legacy) graph to visualize landmarks over the source video in real time for demo. Optionally records the overlay and logs hand landmarks (`hands_1.csv`) together with detection confidences.
- `2.extract_landmarks.py` - Uses the **MediaPipe Tasks** (morden) models (`weights/face_landmarker.task`, `weights/hand_landmarker.task`) to extract normalized `(x, y, z)` coordinates per frame, persist them to CSV (`face.csv`, `hands_2.csv`), and save annotated frames for offline review. Supports filtering subsets of facial keypoints defined in `keypoints.json`.
- `3.merge_hands_csv.py` - Compares the two hand CSVs and keeps the landmarks whose detector reported the highest confidence for each hand, producing a consolidated file for downstream analysis.
- `check_extracted_hand_points.py` - Quick visual QA: replays a video and plots merged hand points (typically `merged_hands.csv`) using the same color palette as the extraction scripts.

## Tech Stack
- Python 3.10+
- MediaPipe Solutions and MediaPipe Tasks
- OpenCV, NumPy, pandas, and Matplotlib

## Keypoint Detection Method
All landmark detection is powered by Google MediaPipe:
- **Hands & Face (legacy path):** `mp.solutions.hands.Hands` and `mp.solutions.face_mesh.FaceMesh`, ideal for rapid visualization.
- **Hands & Face (tasks path):** `mp.tasks.vision.HandLandmarker` and `mp.tasks.vision.FaceLandmarker`, driven by lightweight `.task` models located in `weights/`. These models run per frame, output normalized coordinates, and expose detector confidence scores that the merge step relies on.

## Environment
1. Install Python 3.10+ (matching the versions tested in `requirements.txt`).
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate          # Windows PowerShell
   pip install -r requirements.txt
   ```
3. Download the official MediaPipe Task weights and place them in `weights/` (filenames must match the defaults in the scripts).  
- `face_landmarker.task` - https://developers.google.com/mediapipe/solutions/vision/face_landmarker  
- `hand_landmarker.task` - https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

## Analysis Modules (`src/analysis`)
- `analyze_distant.py` - Offline analytics for exported CSVs: Calculate and plot the Euclidean distance of given pairs of point. 
- `face_points_distant_demo.py` - Realtime analytic (face only): Calculate and plot the Euclidean distance of given pairs of points alongside over frames.
