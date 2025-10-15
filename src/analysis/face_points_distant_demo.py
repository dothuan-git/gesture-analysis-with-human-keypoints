import os
import cv2
import csv
import json
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Set, Dict
from collections import deque

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils import *


# Configuration constants
CONFIG = {
    'VIDEO_PATH': 'assets/IMG_0004.MP4',
    'OUTPUT_PATH': '',
    'KEYPOINTS_FILTER': [
        'lipsUpperOuter', 'lipsLowerOuter',
    ],  # List of keys from keypoints.json
    'LANDMARK_PAIRS': [
        (0, 17),
    ],  # List of (landmark_id1, landmark_id2) pairs to measure distance
    'PLOT_HISTORY_LENGTH': 300,  # Number of frames to show in plot
    'PLOT_SIZE': (1200, 600),  # Plot size in pixels (width, height)
    'SAVE_FRAMES_WITH_PLOT': False,  # Save frames with plot side-by-side
    'DRAW_KEYPOINT_IDS': False,
}


def load_keypoint_indices(filter_keys: List[str]) -> Optional[Set[int]]:
    """Load keypoint indices from keypoints.json based on filter keys."""
    if not filter_keys:
        return None
    
    try:
        with open('keypoints.json', 'r') as f:
            keypoints_data = json.load(f)
        
        indices = set()
        for key in filter_keys:
            if key in keypoints_data:
                if key in ['rightHand', 'leftHand']:
                    continue
                indices.update(keypoints_data[key])
            else:
                print(f"Warning: Key '{key}' not found in keypoints.json")
        
        return indices if indices else None
    except FileNotFoundError:
        print("Warning: keypoints.json not found. Processing all keypoints.")
        return None
    except json.JSONDecodeError:
        print("Warning: Invalid JSON in keypoints.json. Processing all keypoints.")
        return None


def initialize_video_capture() -> Tuple[cv2.VideoCapture, Tuple[int, int]]:
    """Initialize video capture and return capture object and frame dimensions."""
    cap = cv2.VideoCapture(CONFIG['VIDEO_PATH'])
    if not cap.isOpened():
        raise RuntimeError("Failed to open video file")
    check_metadata(cap)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, (width, height)


def calculate_landmark_distances(landmarks, landmark_pairs: List[Tuple[int, int]], 
                                 width: int, height: int) -> Dict[Tuple[int, int], float]:
    """Calculate Euclidean distances between landmark pairs in pixel coordinates."""
    distances = {}
    
    for pair in landmark_pairs:
        id1, id2 = pair
        if id1 < len(landmarks) and id2 < len(landmarks):
            lm1 = landmarks[id1]
            lm2 = landmarks[id2]
            
            # Convert normalized coordinates to pixel coordinates
            x1, y1 = lm1.x * width, lm1.y * height
            x2, y2 = lm2.x * width, lm2.y * height
            
            # Calculate Euclidean distance
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances[pair] = distance
        else:
            distances[pair] = 0.0
            
    return distances


def create_distance_plot(distance_history: Dict[Tuple[int, int], deque], 
                        frame_idx: int) -> np.ndarray:
    """Create a matplotlib plot of landmark distances over time."""
    # Set the backend to Agg to avoid display issues
    plt.switch_backend('Agg')
    
    plot_width, plot_height = CONFIG['PLOT_SIZE']
    fig, ax = plt.subplots(figsize=(plot_width/100, plot_height/100), dpi=100)
    
    # Plot each landmark pair
    colors = plt.cm.tab10(np.linspace(0, 1, len(distance_history)))
    for i, (pair, history) in enumerate(distance_history.items()):
        frames = list(range(max(0, frame_idx - len(history) + 1), frame_idx + 1))
        ax.plot(frames, list(history), label=f'Pair {pair[0]}-{pair[1]}', 
                color=colors[i], linewidth=2)
    
    ax.set_xlabel('Frame', fontsize=10)
    ax.set_ylabel('Distance (pixels)', fontsize=10)
    ax.set_title('Landmark Distances Over Time', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Auto-scale y-axis based on data
    ax.relim()
    ax.autoscale_view()
    
    # Convert plot to image
    fig.canvas.draw()
    # Use buffer_rgba() instead of tostring_rgb()
    buf = fig.canvas.buffer_rgba()
    plot_img = np.asarray(buf)
    plt.close(fig)
    
    # Convert RGBA to BGR for OpenCV (drop alpha channel)
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
    
    return plot_img


def extract_and_write_distances(frame_index: int, distances: Dict[Tuple[int, int], float], 
                                csv_path: str) -> None:
    """Write landmark distances to CSV."""
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            header = ['frame'] + [f'dist_{pair[0]}_{pair[1]}' for pair in distances.keys()]
            writer.writerow(header)
        row = [frame_index] + list(distances.values())
        writer.writerow(row)


def process_frame(frame: np.ndarray, frame_idx: int, width: int, height: int, 
                 output_dir: str, keypoint_indices: Optional[Set[int]],
                 distance_history: Dict[Tuple[int, int], deque]) -> Tuple[np.ndarray, np.ndarray]:
    """Process a frame for face detection, visualization, and distance calculation."""
    annotated_frame = frame.copy()
    mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw_ids = CONFIG.get('DRAW_KEYPOINT_IDS', False)

    # Face detection and processing
    face_options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="weights/face_landmarker.task"),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    
    with mp.tasks.vision.FaceLandmarker.create_from_options(face_options) as landmarker:
        face_result = landmarker.detect(mediapipe_image)
        
        if face_result.face_landmarks:
            raw = face_result.face_landmarks[0]
            landmarks = getattr(raw, 'landmarks', raw)
            
            # Calculate distances between landmark pairs
            distances = calculate_landmark_distances(landmarks, CONFIG['LANDMARK_PAIRS'], 
                                                     width, height)
            
            # Update distance history
            for pair, distance in distances.items():
                if pair not in distance_history:
                    distance_history[pair] = deque(maxlen=CONFIG['PLOT_HISTORY_LENGTH'])
                distance_history[pair].append(distance)
            
            # Save distances to CSV
            distances_csv = f'{output_dir}/landmark_distances.csv'
            extract_and_write_distances(frame_idx, distances, csv_path=distances_csv)
            
            # Visualize landmarks (filtered or all)
            if keypoint_indices is not None:
                sorted_indices = sorted(keypoint_indices)
                filtered_xy = np.array([[landmarks[i].x * width, landmarks[i].y * height] 
                                       for i in sorted_indices],
                                      dtype=np.float32)
                annotated_frame = keypoints_visualizer(
                    annotated_frame, filtered_xy, draw_ids=draw_ids, landmark_ids=sorted_indices
                )
            else:
                face_xy = np.array([[lm.x * width, lm.y * height] for lm in landmarks],
                                  dtype=np.float32)
                annotated_frame = keypoints_visualizer(annotated_frame, face_xy, draw_ids=draw_ids)
            
            # Draw lines between measured pairs
            for pair in CONFIG['LANDMARK_PAIRS']:
                id1, id2 = pair
                if id1 < len(landmarks) and id2 < len(landmarks):
                    pt1 = (int(landmarks[id1].x * width), int(landmarks[id1].y * height))
                    pt2 = (int(landmarks[id2].x * width), int(landmarks[id2].y * height))
                    cv2.line(annotated_frame, pt1, pt2, (0, 0, 255), 1)
                    
                    # Draw distance text at midpoint
                    mid_x = (pt1[0] + pt2[0]) // 2
                    mid_y = (pt1[1] + pt2[1]) // 2
                    distance_text = f"{distances[pair]:.1f}"
                    cv2.putText(annotated_frame, distance_text, (mid_x + 5, mid_y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Create distance plot
    plot_img = create_distance_plot(distance_history, frame_idx)
    
    # Add frame index text
    cv2.putText(annotated_frame, f"Frame: {frame_idx}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    return annotated_frame, plot_img


def main():
    """Main function to process video, extract landmarks, calculate distances, and visualize."""
    # Create output directory
    output_dir = CONFIG["OUTPUT_PATH"]
    if not os.path.exists(output_dir) or len(output_dir) == 0:
        root_path = f'workspaces/{CONFIG["VIDEO_PATH"].split("/")[-1].split(".")[0]}'
        output_dir = create_incremented_dir(root_path, 'runs')

    print(f"Output directory: {output_dir}")
    print(f"Measuring distances for pairs: {CONFIG['LANDMARK_PAIRS']}")

    # Load keypoint filter indices
    keypoint_indices = load_keypoint_indices(CONFIG.get('KEYPOINTS_FILTER', []))
    if keypoint_indices:
        print(f"Filtering {len(keypoint_indices)} keypoints: {sorted(keypoint_indices)}")
    else:
        print("Processing all keypoints")

    # Initialize distance history
    distance_history: Dict[Tuple[int, int], deque] = {}

    try:
        cap, (width, height) = initialize_video_capture()
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Stream ended.")
                break

            annotated_frame, plot_img = process_frame(frame, frame_idx, width, height, 
                                                     output_dir, keypoint_indices, 
                                                     distance_history)

            # Combine annotated frame and plot side by side without stretching
            # Get plot dimensions
            plot_h, plot_w = plot_img.shape[:2]
            frame_h, frame_w = annotated_frame.shape[:2]
            
            # Create a combined canvas with max height
            combined_height = max(frame_h, plot_h)
            combined_width = frame_w + plot_w
            combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
            
            # Place annotated frame on the left (centered vertically if needed)
            y_offset_frame = (combined_height - frame_h) // 2
            combined_frame[y_offset_frame:y_offset_frame + frame_h, :frame_w] = annotated_frame
            
            # Place plot on the right (centered vertically if needed)
            y_offset_plot = (combined_height - plot_h) // 2
            combined_frame[y_offset_plot:y_offset_plot + plot_h, frame_w:frame_w + plot_w] = plot_img

            # Save combined frame (with plot)
            if CONFIG['SAVE_FRAMES_WITH_PLOT']:
                frames_plot_dir = f'{output_dir}/frames_with_plot'
                create_folder_if_not_exist(frames_plot_dir)
                cv2.imwrite(f'{frames_plot_dir}/{frame_idx:06d}.jpg', combined_frame)

            cv2.imshow('Press Q to quit...', combined_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()