import os
import cv2
import csv
import json
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Set, Dict
from collections import deque
import wave

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.analysis.face_points_distant_demo import *
from src.utils import *


# Configuration constants
CONFIG = {
    'VIDEO_PATH': 'workspaces/IMG_0018/chunks_001/video/segment1.mp4',
    'AUDIO_PATH': 'workspaces/IMG_0018/chunks_001/audio/segment1.wav',
    'OUTPUT_PATH': '',
    'KEYPOINTS_FILTER': [
        'lipsUpperOuter', 'lipsLowerOuter',
    ],  # List of keys from keypoints.json
    'LANDMARK_PAIRS': [
        (0, 17),
    ],  # List of (landmark_id1, landmark_id2) pairs to measure distance
    'PLOT_HISTORY_LENGTH': 300,  # Number of frames to show in plot
    'PLOT_SIZE': (16, 10),  # Plot size in pixels (width, height) - increased for waveform
    'SAVE_FRAMES_WITH_PLOT': False,  # Save frames with plot side-by-side
    'DRAW_KEYPOINT_IDS': False,
}


def load_audio_waveform(audio_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Load audio waveform from WAV file."""
    if not os.path.exists(audio_path):
        print(f"Warning: Audio file not found: {audio_path}")
        return None, None
    
    try:
        with wave.open(audio_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            n_frames = wav_file.getnframes()
            sample_width = wav_file.getsampwidth()
            
            print(f"Audio info: {sample_rate} Hz, {n_channels} channel(s), {sample_width} bytes/sample")
            
            audio_data = wav_file.readframes(n_frames)
            
            # Convert to numpy array based on sample width
            if sample_width == 1:
                dtype = np.uint8
            elif sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                print(f"Warning: Unsupported sample width: {sample_width}")
                return None, None
            
            audio_array = np.frombuffer(audio_data, dtype=dtype)
            
            # If stereo, convert to mono by averaging channels
            if n_channels == 2:
                audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(dtype)
                print("Converted stereo to mono")
            
            return audio_array, sample_rate
            
    except Exception as e:
        print(f"Warning: Error loading audio file: {e}")
        return None, None


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


def create_distance_and_waveform_plot(distance_history: Dict[Tuple[int, int], deque], 
                                      frame_idx: int,
                                      audio_data: Optional[np.ndarray],
                                      sample_rate: Optional[int],
                                      fps: float,
                                      video_name: str) -> np.ndarray:
    """Create a matplotlib plot with distance plot and waveform."""
    # Set the backend to Agg to avoid display issues
    plt.switch_backend('Agg')
    
    plot_width, plot_height = CONFIG['PLOT_SIZE']
    fig = plt.figure(figsize=(plot_width, plot_height), dpi=100)
    
    # Create two subplots: distance plot on top, waveform on bottom
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    
    # Plot 1: Distance plot
    colors = plt.cm.tab10(np.linspace(0, 1, len(distance_history)))
    for i, (pair, history) in enumerate(distance_history.items()):
        frames = list(range(max(0, frame_idx - len(history) + 1), frame_idx + 1))
        ax1.plot(frames, list(history), label=f'Pair {pair[0]}-{pair[1]}', marker='o', 
                color=colors[i], linewidth=2)
    
    ax1.set_xlabel('Frame', fontsize=10)
    ax1.set_ylabel('Distance (pixels)', fontsize=10)
    ax1.set_title(f'Landmark Distances Over Time - {video_name}', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.relim()
    ax1.autoscale_view()
    
    # Plot 2: Waveform
    if audio_data is not None and sample_rate is not None:
        # Calculate current time in video
        current_time = frame_idx / fps
        
        # Show full audio waveform
        time_array = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
        
        # Plot full waveform
        ax2.plot(time_array, audio_data, color='steelblue', linewidth=0.5)
        
        # Add vertical line for current position
        ax2.axvline(x=current_time, color='red', linestyle='--', linewidth=2, 
                   label='Current Frame')
        
        ax2.set_xlabel('Time (seconds)', fontsize=10)
        ax2.set_ylabel('Amplitude', fontsize=10)
        ax2.set_title(f'Audio Waveform (Full Length) - {video_name}', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
    else:
        ax2.text(0.5, 0.5, 'No Audio Available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax2.transAxes, fontsize=14, color='gray')
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    plt.tight_layout()
    
    # Convert plot to image
    fig.canvas.draw()
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
                 distance_history: Dict[Tuple[int, int], deque],
                 audio_data: Optional[np.ndarray],
                 sample_rate: Optional[int],
                 fps: float,
                 video_name: str) -> Tuple[np.ndarray, np.ndarray]:
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
                    cv2.line(annotated_frame, pt1, pt2, (0, 0, 255), 2)
                    
                    # Draw distance text at midpoint
                    mid_x = (pt1[0] + pt2[0]) // 2
                    mid_y = (pt1[1] + pt2[1]) // 2
                    distance_text = f"{distances[pair]:.1f}"
                    cv2.putText(annotated_frame, distance_text, (mid_x + 5, mid_y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Create distance and waveform plot
    plot_img = create_distance_and_waveform_plot(distance_history, frame_idx, 
                                                 audio_data, sample_rate, fps, video_name)
    
    # Add frame index text
    cv2.putText(annotated_frame, f"Frame: {frame_idx}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    return annotated_frame, plot_img


def main():
    """Main function to process video, extract landmarks, calculate distances, and visualize."""
    # Validate input paths
    video_path = CONFIG['VIDEO_PATH']
    audio_path = CONFIG['AUDIO_PATH']
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not os.path.exists(audio_path):
        print(f"Warning: Audio file not found: {audio_path}")
        print("Continuing without audio...")
    
    # Create output directory
    output_dir = CONFIG["OUTPUT_PATH"]
    if not os.path.exists(output_dir) or len(output_dir) == 0:
        root_path = f'workspaces/{video_path.split("/")[-1].split(".")[0]}'
        output_dir = create_incremented_dir(root_path, 'runs')

    print(f"Output directory: {output_dir}")
    print(f"Video path: {video_path}")
    print(f"Audio path: {audio_path}")
    print(f"Measuring distances for pairs: {CONFIG['LANDMARK_PAIRS']}")

    # Load audio waveform
    print("\nLoading audio...")
    audio_data, sample_rate = load_audio_waveform(audio_path)
    if audio_data is not None:
        print(f"Audio loaded: {len(audio_data)} samples at {sample_rate} Hz")
        print(f"Audio duration: {len(audio_data) / sample_rate:.2f} seconds")
    else:
        print("No audio available or loading failed")

    # Load keypoint filter indices
    keypoint_indices = load_keypoint_indices(CONFIG.get('KEYPOINTS_FILTER', []))
    if keypoint_indices:
        print(f"Filtering {len(keypoint_indices)} keypoints: {sorted(keypoint_indices)}")
    else:
        print("Processing all keypoints")

    # Initialize distance history
    distance_history: Dict[Tuple[int, int], deque] = {}
    
    # Extract video name from path
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    try:
        cap, (width, height), fps = initialize_video_capture(video_path)
        print(f"Video: {width}x{height} @ {fps} fps")
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        print(f"Video duration: {video_duration:.2f} seconds ({total_frames} frames)")
        
        # Check if audio and video durations match
        if audio_data is not None:
            audio_duration = len(audio_data) / sample_rate
            duration_diff = abs(audio_duration - video_duration)
            if duration_diff > 0.1:  # More than 100ms difference
                print(f"Warning: Audio ({audio_duration:.2f}s) and video ({video_duration:.2f}s) durations differ by {duration_diff:.2f}s")
        
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Stream ended.")
                break

            annotated_frame, plot_img = process_frame(frame, frame_idx, width, height, 
                                                     output_dir, keypoint_indices, 
                                                     distance_history, audio_data, 
                                                     sample_rate, fps, video_name)

            # Combine annotated frame and plot depending on frame orientation
            combined_frame = combine_frame_and_plot(annotated_frame, plot_img)

            # Save combined frame (with plot)
            if CONFIG['SAVE_FRAMES_WITH_PLOT']:
                frames_plot_dir = f'{output_dir}/frames_with_plot'
                create_folder_if_not_exist(frames_plot_dir)
                cv2.imwrite(f'{frames_plot_dir}/{frame_idx:06d}.jpg', combined_frame)

            resized_for_display, _, _ = fit_to_screen(combined_frame)
            cv2.imshow('Press Q to quit...', resized_for_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1

        # Save the last plot
        if distance_history:
            last_plot = create_distance_and_waveform_plot(distance_history, frame_idx - 1,
                                                         audio_data, sample_rate, fps, video_name)
            cv2.imwrite(f'{output_dir}/last_plot.jpg', last_plot)
            print(f"\nProcessed {frame_idx} frames")
            print(f"Results saved to: {output_dir}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()