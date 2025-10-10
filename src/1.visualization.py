import os
import cv2
import csv
import numpy as np
import mediapipe as mp
import utils as utils
from typing import Optional, Tuple

# Configuration constants
CONFIG = {
    'VIDEO_PATH': 'assets/IMG_0004.MP4',
    'REAL_TIME': False,
    'SAVE_VIDEO': False,
    'SAVE_FRAMES': False,
    'SHOW_FACE_MESH': False
}

# Mediapipe drawing styles
DRAWING_STYLES = {
    'face_landmark': mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
    'face_connection': mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1),

    'left_hand_landmark': mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
    'left_hand_connection': mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1),

    'right_hand_landmark': mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
    'right_hand_connection': mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1),
}


def initialize_mediapipe() -> Tuple[mp.solutions.hands.Hands, mp.solutions.face_mesh.FaceMesh]:
    """Initialize Mediapipe hands and face mesh detectors."""
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return hands, face_mesh


def initialize_video_capture() -> Tuple[cv2.VideoCapture, Tuple[int, int], float]:
    """Initialize video capture from file or camera."""
    cap = cv2.VideoCapture(0 if CONFIG['REAL_TIME'] else CONFIG['VIDEO_PATH'])
    if not cap.isOpened():
        raise RuntimeError("Failed to open video source")
    
    if CONFIG['REAL_TIME']:
        utils.set_res(cap)
    else:
        utils.check_metadata(cap)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return cap, (width, height), fps


def initialize_video_writer(width: int, height: int, fps: float) -> Optional[cv2.VideoWriter]:
    """Initialize video writer if saving is enabled."""
    if not CONFIG['SAVE_VIDEO']:
        return None
    output_path = utils.get_video_path(prefix='visualized', fps=int(fps))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def extract_and_write_hand_landmarks(frame_index: int, hands_res, csv_path: str) -> None:
    """Extract and write hand landmarks to CSV with confidence scores, using highest confidence for same hand type."""
    right_landmarks, left_landmarks = [], []
    right_max_score, left_max_score = -1.0, -1.0
    right_idx, left_idx = -1, -1

    if hands_res.multi_handedness and hands_res.multi_hand_landmarks:
        for i, hand in enumerate(hands_res.multi_handedness):
            score = hand.classification[0].score
            label = hand.classification[0].label.lower()
            if label == 'right' and score > right_max_score:
                right_max_score, right_idx = score, i
            elif label == 'left' and score > left_max_score:
                left_max_score, left_idx = score, i

    if right_idx >= 0 and right_idx < len(hands_res.multi_hand_landmarks):
        right_landmarks = hands_res.multi_hand_landmarks[right_idx]
    if left_idx >= 0 and left_idx < len(hands_res.multi_hand_landmarks):
        left_landmarks = hands_res.multi_hand_landmarks[left_idx]

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
                if landmarks and i < len(landmarks.landmark):
                    lm = landmarks.landmark[i]
                    row += [lm.x, lm.y, lm.z]
                else:
                    row += [0.0, 0.0, 0.0]
        writer.writerow(row)


def process_frame(frame: np.ndarray, hands: mp.solutions.hands.Hands,
                  face_mesh: mp.solutions.face_mesh.FaceMesh, frame_idx: int,
                  output_dir: str) -> np.ndarray:
    """Process a single frame for hand and face detection, visualization, and CSV writing."""
    h, w, _ = frame.shape
    # frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False

    hands_res = hands.process(rgb)
    face_res = face_mesh.process(rgb)
    rgb.flags.writeable = True
    annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Write hand landmarks to CSV
    hand_csv = f'{output_dir}/hands_1.csv'
    utils.create_folder_if_not_exist(output_dir)
    extract_and_write_hand_landmarks(frame_idx, hands_res, hand_csv)

    # Draw hand landmarks
    if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
        for hand_landmarks, handedness in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
            label = handedness.classification[0].label
            lm_style = DRAWING_STYLES[f'{"left" if label == "Left" else "right"}_hand_landmark']
            conn_style = DRAWING_STYLES[f'{"left" if label == "Left" else "right"}_hand_connection']
            mp.solutions.drawing_utils.draw_landmarks(
                annotated, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                landmark_drawing_spec=lm_style, connection_drawing_spec=conn_style
            )

    # Draw face landmarks
    if face_res.multi_face_landmarks:
        for face_landmarks in face_res.multi_face_landmarks:
            connections = mp.solutions.face_mesh.FACEMESH_TESSELATION if CONFIG['SHOW_FACE_MESH'] else None
            mp.solutions.drawing_utils.draw_landmarks(
                annotated, face_landmarks, connections,
                landmark_drawing_spec=DRAWING_STYLES['face_landmark'],
                connection_drawing_spec=DRAWING_STYLES['face_connection'] if connections else None
            )

    # Add frame index text
    cv2.putText(annotated, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    return annotated


def main():
    """Main function to process video and visualize landmarks."""
    try:
        hands, face_mesh = initialize_mediapipe()
        cap, (width, height), fps = initialize_video_capture()
        video_writer = initialize_video_writer(width, height, fps)
        output_dir = f'data/{CONFIG["VIDEO_PATH"].split("/")[-1].split(".")[0]}'

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated = process_frame(frame, hands, face_mesh, frame_idx, output_dir)

            cv2.imshow("Hands + FaceMesh", annotated)
            if video_writer:
                video_writer.write(annotated)
            if CONFIG['SAVE_FRAMES']:
                frames_dir = f'{output_dir}/frames_1'
                utils.create_folder_if_not_exist(frames_dir)
                cv2.imwrite(f'{frames_dir}/{frame_idx}.jpg', annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        if video_writer:
            video_writer.release()
            print(f"Saved output video to: {utils.get_video_path(prefix='visualized', fps=int(fps))}")
        cv2.destroyAllWindows()
        hands.close()
        face_mesh.close()


if __name__ == "__main__":
    main()