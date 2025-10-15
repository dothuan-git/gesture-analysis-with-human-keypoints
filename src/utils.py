# utils.py
import cv2
import os
import re
import json
import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


def check_metadata(cap, log=True):
    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        length_in_seconds = frame_count / fps

        if log:
            print("-----------------------------------------")
            print(f"Width: {width}")
            print(f"Height: {height}")
            print(f"FPS: {fps}")
            print(f"Frame count: {frame_count}")
            print(f"Length (in seconds): {length_in_seconds:.2f}")
            print("-----------------------------------------")

        return width, height, fps, frame_count


def set_res(cap, x=1080, y=720, fps=60):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    cap.set(cv2.CAP_PROP_FPS, fps)
    return str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def get_current_time():
    now = datetime.datetime.now()
    date_only = str(now.date())
    time_only = str(now.time()).split(".")[0].replace(":", ".")
    return date_only, time_only


def get_video_path(prefix, fps):
    date_now, time_now = get_current_time()
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    video_path = f"{prefix}/{date_now}_{time_now}[{fps}fps].avi"
    return video_path


def create_folder_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")


def create_incremented_dir(root: str | Path, prefix: str) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    patt = re.compile(rf"^{re.escape(prefix)}{re.escape('_')}(\d{{{3}}})$")

    # Find current max index
    current_max = 0
    for p in root.iterdir():
        if p.is_dir():
            m = patt.match(p.name)
            if m:
                current_max = max(current_max, int(m.group(1)))

    # Create next available dir; loop handles rare race with another process
    idx = current_max + 1
    while True:
        candidate = root / f"{prefix}_{idx:0{3}d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            idx += 1


def get_latest_or_create(root: str | Path, prefix: str) -> Path:
    root = Path(root)
    patt = re.compile(rf"^{re.escape(prefix)}{re.escape('_')}(\d{{{3}}})$")

    # If root doesn't exist or has no matching dirs, create the first/next one.
    if not root.exists():
        return create_incremented_dir(root, prefix)

    latest_path = None
    latest_idx = 0

    for p in root.iterdir():
        if p.is_dir():
            m = patt.match(p.name)
            if m:
                idx = int(m.group(1))
                if idx > latest_idx:
                    latest_idx = idx
                    latest_path = p

    if latest_path is None:
        # No matching directories found -> create one
        return create_incremented_dir(root, prefix)

    return latest_path


def load_hand_connections(json_path='hand_connections.json'):
    """Load hand connections from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return [tuple(conn) for conn in data['connections']]


def keypoints_visualizer(frame, keypoints_xy, edges=None, draw_ids=False, color=(0, 255, 0), landmark_ids=None):
    """
    Visualize keypoints and skeleton connections using pure OpenCV.
    """
    annotated_frame = frame.copy()
    
    # Handle both numpy array and sv.KeyPoints object
    if hasattr(keypoints_xy, 'xy'):
        # It's a sv.KeyPoints object
        xy = keypoints_xy.xy[0]  # shape: (num_points, 2)
    else:
        # It's already a numpy array
        xy = keypoints_xy
    
    # Ensure xy is the right shape
    if len(xy.shape) == 3:
        xy = xy[0]  # Remove batch dimension if present
    
    # Draw skeleton edges first (so they appear behind points)
    if edges is not None:
        skeleton_color = (255, 255, 255)
        for start_idx, end_idx in edges:
            if start_idx < len(xy) and end_idx < len(xy):
                start_point = tuple(xy[start_idx].astype(int))
                end_point = tuple(xy[end_idx].astype(int))
                cv2.line(annotated_frame, start_point, end_point, skeleton_color, thickness=1)
    
    # Draw keypoints (circles)
    for idx, (x, y) in enumerate(xy):
        center = (int(x), int(y))
        cv2.circle(annotated_frame, center, radius=3, color=color, thickness=-1)
    
    # Draw IDs above each point
    if draw_ids:
        text_color = (255, 255, 0)
        for idx, (x, y) in enumerate(xy):
            # Use original landmark ID if provided, otherwise use array index
            label = str(landmark_ids[idx]) if landmark_ids is not None else str(idx)
            cv2.putText(
                annotated_frame,
                label,
                (int(x), int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                text_color,
                1,
                cv2.LINE_AA
            )
    
    return annotated_frame


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())
    return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
    face_blendshapes_names = [cat.category_name for cat in face_blendshapes]
    face_blendshapes_scores = [cat.score for cat in face_blendshapes]
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores)
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()
