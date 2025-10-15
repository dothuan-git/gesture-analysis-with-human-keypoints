import os
import cv2
import csv
import json
import supervision as sv
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List, Set
import utils as utils

# Configuration constants
CONFIG = {
    'VIDEO_PATH': 'assets/GX010016_1080_120fps.MP4',
    'DETECT_FACE': True,
    'DETECT_HANDS': True,
    'SAVE_FRAMES': True,
    'KEYPOINTS_FILTER': [
        'lipsUpperOuter', 'lipsLowerOuter',
    ],  # List of keys from keypoints.json, e.g., ['lipsUpperOuter', 'lipsLowerOuter']
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
                    # Skip hand keypoints as they're handled separately
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


def filter_landmarks(landmarks, keypoint_indices: Optional[Set[int]]):
    """Filter landmarks based on keypoint indices."""
    if keypoint_indices is None:
        return landmarks
    
    filtered = []
    for i, lm in enumerate(landmarks):
        if i in keypoint_indices:
            filtered.append(lm)
    return filtered


def initialize_video_capture() -> Tuple[cv2.VideoCapture, Tuple[int, int]]:
    """Initialize video capture and return capture object and frame dimensions."""
    cap = cv2.VideoCapture(CONFIG['VIDEO_PATH'])
    if not cap.isOpened():
        raise RuntimeError("Failed to open video file")
    utils.check_metadata(cap)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, (width, height)


def extract_and_write_face_landmarks(frame_index: int, face_result, csv_path: str, 
                                     keypoint_indices: Optional[Set[int]] = None) -> None:
    """Extract up to one face's normalized x,y,z landmarks and append to CSV."""
    raw = face_result.face_landmarks or []
    landmarks = getattr(raw[0], 'landmarks', raw[0]) if raw else []
    
    # Filter landmarks if keypoint_indices is provided
    if keypoint_indices is not None:
        filtered_landmarks = []
        filtered_indices = []
        for i, lm in enumerate(landmarks):
            if i in keypoint_indices:
                filtered_landmarks.append(lm)
                filtered_indices.append(i)
        landmarks = filtered_landmarks
        landmark_indices = filtered_indices
    else:
        landmark_indices = list(range(len(landmarks)))
    
    num_landmarks = len(landmarks)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            header = ['frame'] + [f"face_{idx}_{coord}" 
                                 for idx in landmark_indices 
                                 for coord in ['x', 'y', 'z']]
            writer.writerow(header)
        row = [frame_index] + [coord for lm in landmarks for coord in [lm.x, lm.y, lm.z]]
        writer.writerow(row)


def extract_and_write_hand_landmarks(frame_index: int, hand_result, csv_path: str) -> None:
    """Extract up to two hands' normalized x,y,z landmarks and confidence scores, append to CSV."""
    right_landmarks, left_landmarks = [], []
    right_max_score, left_max_score = -1.0, -1.0
    right_idx, left_idx = -1, -1

    for i, hand in enumerate(hand_result.handedness):
        score = hand[0].score
        handedness = hand[0].category_name.lower()
        
        if handedness == 'left' and score > right_max_score:  # Treat 'left' as 'right'
            right_max_score, right_idx = score, i
        elif handedness == 'right' and score > left_max_score:  # Treat 'right' as 'left'
            left_max_score, left_idx = score, i

    if right_idx >= 0 and right_idx < len(hand_result.hand_landmarks):
        right_landmarks = hand_result.hand_landmarks[right_idx]
    if left_idx >= 0 and left_idx < len(hand_result.hand_landmarks):
        left_landmarks = hand_result.hand_landmarks[left_idx]

    num_landmarks_per_hand = 21
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            header = ['frame', 'right_confidence', 'left_confidence'] + \
                     [f"{hand}_{i}_{coord}" for hand in ['right', 'left']
                      for i in range(num_landmarks_per_hand) for coord in ['x', 'y', 'z']]
            writer.writerow(header)

        row = [frame_index, right_max_score if right_max_score >= 0 else 0.0,
               left_max_score if left_max_score >= 0 else 0.0]
        for landmarks in [right_landmarks, left_landmarks]:
            for i in range(num_landmarks_per_hand):
                if landmarks and i < len(landmarks):
                    lm = landmarks[i]
                    row += [lm.x, lm.y, lm.z]
                else:
                    row += [0.0, 0.0, 0.0]
        writer.writerow(row)


def process_frame(frame: np.ndarray, frame_idx: int, width: int, height: int, 
                 output_dir: str, keypoint_indices: Optional[Set[int]] = None) -> np.ndarray:
    """Process a frame for face and hand detection, visualization, and data extraction."""
    # frame = cv2.flip(frame, 1)
    annotated_frame = frame.copy()
    mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw_ids = CONFIG.get('DRAW_KEYPOINT_IDS', False)

    # Face detection and processing
    if CONFIG['DETECT_FACE']:
        face_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="weights/face_landmarker.task"),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        with mp.tasks.vision.FaceLandmarker.create_from_options(face_options) as landmarker:
            face_result = landmarker.detect(mediapipe_image)
            face_csv = f'{output_dir}/face.csv'
            extract_and_write_face_landmarks(frame_idx, face_result, csv_path=face_csv, 
                                            keypoint_indices=keypoint_indices)
            
            # Visualize filtered keypoints with original IDs
            if keypoint_indices is not None and face_result.face_landmarks:
                raw = face_result.face_landmarks[0]
                landmarks = getattr(raw, 'landmarks', raw)
                sorted_indices = sorted(keypoint_indices)
                filtered_xy = np.array([[landmarks[i].x * width, landmarks[i].y * height] 
                                       for i in sorted_indices],
                                      dtype=np.float32).reshape(1, -1, 2)
                filtered_confidence = np.ones((1, len(filtered_xy[0])))
                face_keypoints = sv.KeyPoints(xy=filtered_xy, confidence=filtered_confidence)
                # Pass original landmark IDs to visualizer
                annotated_frame = utils.keypoints_visualizer(
                    annotated_frame, face_keypoints, draw_ids=draw_ids, landmark_ids=sorted_indices
                )
            else:
                # Visualize all keypoints
                face_keypoints = sv.KeyPoints.from_mediapipe(face_result, (width, height))
                annotated_frame = utils.keypoints_visualizer(annotated_frame, face_keypoints)

    # Hand detection and processing
    if CONFIG['DETECT_HANDS']:
        hand_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="weights/hand_landmarker.task"),
            num_hands=2,
            min_hand_detection_confidence=0.5
        )
        with mp.tasks.vision.HandLandmarker.create_from_options(hand_options) as handmarker:
            hand_result = handmarker.detect(mediapipe_image)
            hand_csv = f'{output_dir}/hands_2.csv'
            extract_and_write_hand_landmarks(frame_idx, hand_result, csv_path=hand_csv)

            if hand_result.hand_landmarks:
                for i, landmarks in enumerate(hand_result.hand_landmarks):
                    handedness = hand_result.handedness[i][0].display_name.lower()
                    color = sv.Color.RED if handedness == "left" else sv.Color.GREEN  # Swap colors
                    keypoints_xy = np.array([[lm.x * width, lm.y * height] for lm in landmarks],
                                           dtype=np.float32).reshape(1, -1, 2)
                    hand_keypoints = sv.KeyPoints(xy=keypoints_xy, confidence=np.ones((1, len(landmarks))))
                    annotated_frame = utils.keypoints_visualizer(annotated_frame, hand_keypoints, draw_ids=draw_ids, color=color)

    # Add frame index text
    cv2.putText(annotated_frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    return annotated_frame


def main():
    """Main function to process video, extract landmarks, and save annotated frames."""
    # Create output directory
    root_path = f'workspaces/{CONFIG["VIDEO_PATH"].split("/")[-1].split(".")[0]}'
    output_dir = utils.create_incremented_dir(root_path, 'runs')

    # Load keypoint filter indices
    keypoint_indices = load_keypoint_indices(CONFIG.get('KEYPOINTS_FILTER', []))
    if keypoint_indices:
        print(f"Filtering {len(keypoint_indices)} keypoints: {sorted(keypoint_indices)}")
    else:
        print("Processing all keypoints")

    try:
        cap, (width, height) = initialize_video_capture()
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Stream ended.")
                break

            annotated_frame = process_frame(frame, frame_idx, width, height, output_dir, keypoint_indices)

            # Save annotated frame
            if CONFIG['SAVE_FRAMES']:
                frames_dir = f'{output_dir}/frames_2'
                utils.create_folder_if_not_exist(frames_dir)
                cv2.imwrite(f'{frames_dir}/{frame_idx}.jpg', annotated_frame)

            cv2.imshow('Press Q to quit...', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()