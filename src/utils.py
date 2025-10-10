# utils.py
import cv2
import os
import datetime
import supervision as sv
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
    

def get_current_time():
    now = datetime.datetime.now()
    date_only = str(now.date())
    time_only = str(now.time()).split(".")[0].replace(":", ".")
    return date_only, time_only


def get_video_path(prefix, fps):
    date_now, time_now = get_current_time()
    directory = f"data/{prefix}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    video_path = f"{directory}/{date_now}_{time_now}[{fps}fps].avi"
    return video_path


def create_folder_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")


def set_res(cap, x=1080, y=720, fps=60):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    cap.set(cv2.CAP_PROP_FPS, fps)
    return str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def keypoints_visualizer(frame, key_points, edges=None, draw_ids=False, color=sv.Color.GREEN, landmark_ids=None):
    annotated_frame = frame.copy()

    # Draw points thicker in specified color
    vertex_annotator = sv.VertexAnnotator(color=color, radius=2)
    annotated_frame = vertex_annotator.annotate(scene=annotated_frame, key_points=key_points)

    # Draw skeleton thinner in red
    if edges is not None:
        edge_annotator = sv.EdgeAnnotator(color=sv.Color.RED, thickness=1, edges=edges)
        annotated_frame = edge_annotator.annotate(scene=annotated_frame, key_points=key_points)

    # Draw ids above each point
    if draw_ids:
        xy = key_points.xy[0]  # shape: (num_points, 2)
        for idx, (x, y) in enumerate(xy):
            # Use original landmark ID if provided, otherwise use array index
            label = str(landmark_ids[idx]) if landmark_ids is not None else str(idx)
            cv2.putText(
                annotated_frame,
                label,
                (int(x), int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 70, 0),
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
