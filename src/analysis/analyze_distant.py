import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Configuration constants
CONFIG = {
    'CSV_PATH': 'data/IMG_0004/face.csv',
    'FPS': 30.0,  # Frames per second of the video
    'RESOLUTION': (720, 1280),  # (width, height) in pixels - CHANGE THIS TO YOUR VIDEO RESOLUTION
    'POINT_PAIRS': [
        (0, 17),    
    ],
    'POINTS_PREFIX': 'face',  # Prefix used in CSV columns for keypoints
    'ZOOM_MODE': 'time',  # 'frame' or 'time' - determines which mode dict to use
    'TIME_MODE': {
        'ZOOM_DURATION': [617, 622],  # [start_time, end_time] in seconds
        'X_AXIS_MODE': 'time',  # Display time on x-axis
    },
    'FRAME_MODE': {
        'ZOOM_DURATION': [10000, 12000],  # [start_frame, end_frame]
        'X_AXIS_MODE': 'frame',  # Display frame numbers on x-axis
    },
    'PLOT_TITLE': 'Facial Keypoint Distance Over Time',
    'FIGURE_SIZE': (20, 10),
    'SAVE_PLOT': True,
}


def load_keypoints(csv_path: str) -> pd.DataFrame:
    """Load keypoints CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} frames from {csv_path}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    except Exception as e:
        raise Exception(f"Error loading CSV: {e}")


def calculate_euclidean_distance(x1: float, y1: float, z1: float, 
                                 x2: float, y2: float, z2: float) -> float:
    """Calculate 3D Euclidean distance between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)


def calculate_2d_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate 2D Euclidean distance between two points (x, y only)."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def calculate_2d_distance_pixels(x1: float, y1: float, x2: float, y2: float, 
                                  width: int, height: int) -> float:
    """Calculate 2D Euclidean distance in pixels.
    
    Args:
        x1, y1: Normalized coordinates of first point (0-1 range)
        x2, y2: Normalized coordinates of second point (0-1 range)
        width: Video width in pixels
        height: Video height in pixels
    
    Returns:
        Distance in pixels
    """
    # Convert normalized coordinates to pixels
    x1_px = x1 * width
    y1_px = y1 * height
    x2_px = x2 * width
    y2_px = y2 * height
    
    return np.sqrt((x2_px - x1_px)**2 + (y2_px - y1_px)**2)


def extract_point_coords(df: pd.DataFrame, point_id: int, prefix: str, frame_idx: int) -> Tuple[float, float, float]:
    """Extract x, y, z coordinates for a specific point and frame."""
    x_col = f'{prefix}_{point_id}_x'
    y_col = f'{prefix}_{point_id}_y'
    z_col = f'{prefix}_{point_id}_z'
    
    if x_col not in df.columns or y_col not in df.columns or z_col not in df.columns:
        raise ValueError(f"Point {point_id} not found in CSV columns")
    
    x = df.loc[frame_idx, x_col]
    y = df.loc[frame_idx, y_col]
    z = df.loc[frame_idx, z_col]
    
    return x, y, z


def calculate_distance_for_pair(df: pd.DataFrame, point_a: int, point_b: int, 
                                prefix: str, use_3d: bool = True, 
                                resolution: Tuple[int, int] = None) -> List[float]:
    """Calculate distance between two points for all frames.
    
    Args:
        df: DataFrame with keypoint data
        point_a, point_b: Point indices
        prefix: Column prefix
        use_3d: If True, use 3D distance, else 2D
        resolution: (width, height) tuple for pixel conversion. If provided, 2D distances are in pixels.
    """
    distances = []
    
    for frame_idx in range(len(df)):
        try:
            x1, y1, z1 = extract_point_coords(df, point_a, prefix, frame_idx)
            x2, y2, z2 = extract_point_coords(df, point_b, prefix, frame_idx)
            
            if use_3d:
                distance = calculate_euclidean_distance(x1, y1, z1, x2, y2, z2)
            else:
                # If resolution is provided, calculate distance in pixels
                if resolution is not None:
                    width, height = resolution
                    distance = calculate_2d_distance_pixels(x1, y1, x2, y2, width, height)
                else:
                    distance = calculate_2d_distance(x1, y1, x2, y2)
            
            distances.append(distance)
        except Exception as e:
            print(f"Warning: Error at frame {frame_idx} for pair ({point_a}, {point_b}): {e}")
            distances.append(np.nan)
    
    return distances


def get_zoom_frames(fps: float, total_frames: int) -> Tuple[int, int]:
    """Get zoom start and end frames based on ZOOM_MODE."""
    zoom_mode = CONFIG.get('ZOOM_MODE', 'frame').lower()
    
    if zoom_mode == 'time':
        # Use TIME_MODE dict
        time_config = CONFIG.get('TIME_MODE', {})
        zoom_duration = time_config.get('ZOOM_DURATION', [0.0, 1.0])
        zoom_start = int(zoom_duration[0] * fps)
        zoom_end = int(zoom_duration[1] * fps)
    else:
        # Use FRAME_MODE dict
        frame_config = CONFIG.get('FRAME_MODE', {})
        zoom_duration = frame_config.get('ZOOM_DURATION', [0, total_frames])
        zoom_start = zoom_duration[0]
        zoom_end = zoom_duration[1]
    
    # Validate and clamp values
    zoom_start = max(0, min(zoom_start, total_frames - 1))
    zoom_end = max(zoom_start + 1, min(zoom_end, total_frames))
    
    return zoom_start, zoom_end


def get_x_axis_mode() -> str:
    """Get X_AXIS_MODE based on ZOOM_MODE."""
    zoom_mode = CONFIG.get('ZOOM_MODE', 'frame').lower()
    
    if zoom_mode == 'time':
        time_config = CONFIG.get('TIME_MODE', {})
        return time_config.get('X_AXIS_MODE', 'time').lower()
    else:
        frame_config = CONFIG.get('FRAME_MODE', {})
        return frame_config.get('X_AXIS_MODE', 'frame').lower()


def frames_to_time(frames: np.ndarray, fps: float) -> np.ndarray:
    """Convert frame numbers to time in seconds."""
    return frames / fps


def plot_distances(df: pd.DataFrame, prefix: str, point_pairs: List[Tuple[int, int]],
                   fps: float, use_3d: bool = True, output_dir: str = '', 
                   resolution: Tuple[int, int] = None) -> None:
    """Plot distance between point pairs over time with zoomed subplot."""
    fig = plt.figure(figsize=CONFIG['FIGURE_SIZE'])
    
    # Create two subplots: full view and zoomed view
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    
    frames = np.arange(len(df))
    time = frames_to_time(frames, fps)
    
    # Determine x-axis data based on mode
    x_axis_mode = get_x_axis_mode()
    if x_axis_mode == 'frame':
        x_data = frames
        x_label = 'Frame Number'
    else:
        x_data = time
        x_label = 'Time (seconds)'
    
    # Get zoom parameters
    zoom_start, zoom_end = get_zoom_frames(fps, len(df))
    
    if x_axis_mode == 'frame':
        zoom_x_start = zoom_start
        zoom_x_end = zoom_end
    else:
        zoom_x_start = zoom_start / fps
        zoom_x_end = zoom_end / fps
    
    # Determine y-axis label based on mode
    if use_3d:
        y_label = 'Distance (normalized units)'
        dimension = "3D"
    else:
        if resolution is not None:
            y_label = 'Distance (pixels)'
            dimension = "2D (pixels)"
        else:
            y_label = 'Distance (normalized units)'
            dimension = "2D"
    
    for point_a, point_b in point_pairs:
        try:
            distances = calculate_distance_for_pair(df, point_a, point_b, prefix, use_3d, resolution)
            label = f"Point {point_a} ↔ Point {point_b}"
            
            # Plot full view
            ax1.plot(x_data, distances, marker='o', markersize=2, label=label, alpha=0.7)
            
            # Plot zoomed view
            ax2.plot(x_data, distances, marker='o', markersize=3, label=label, alpha=0.7)
            
            # Print statistics
            mean_dist = np.nanmean(distances)
            std_dist = np.nanstd(distances)
            print(f"\n{label}:")
            print(f"  Mean distance: {mean_dist:.6f}")
            print(f"  Std deviation: {std_dist:.6f}")
            print(f"  Min: {np.nanmin(distances):.6f}")
            print(f"  Max: {np.nanmax(distances):.6f}")
            
        except Exception as e:
            print(f"Error plotting pair ({point_a}, {point_b}): {e}")
    
    # Configure full view subplot
    ax1.set_xlabel(x_label, fontsize=11)
    ax1.set_ylabel(y_label, fontsize=11)
    ax1.set_title(f"{CONFIG['PLOT_TITLE']} ({dimension}) - Full View", fontsize=13)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Highlight zoom region on full view
    ax1.axvspan(zoom_x_start, zoom_x_end, alpha=0.2, color='yellow', 
                label=f'Zoom region')
    
    # Configure zoomed view subplot
    ax2.set_xlim(zoom_x_start, zoom_x_end)
    ax2.set_xlabel(x_label, fontsize=11)
    ax2.set_ylabel(y_label, fontsize=11)
    ax2.set_title(f"Zoomed View: Frames {zoom_start}-{zoom_end} ({zoom_x_start:.2f}s - {zoom_x_end:.2f}s)", 
                    fontsize=13)
    
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    suffix = "pixels" if (not use_3d and resolution is not None) else x_axis_mode
    plot_path = f"{output_dir}/distance_plot_{suffix} [{zoom_x_start}-{zoom_x_end}].png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    plt.show()


def analyze_displacement_changes(df: pd.DataFrame, point_pairs: List[Tuple[int, int]], 
                                fps: float, use_3d: bool = True, resolution: Tuple[int, int] = None):
    """Analyze frame-to-frame distance changes (displacement/velocity) with zoomed subplot."""
    fig = plt.figure(figsize=CONFIG['FIGURE_SIZE'])
    
    # Create two subplots: full view and zoomed view
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    
    frames = np.arange(len(df))
    time = frames_to_time(frames, fps)
    
    # Determine x-axis data based on mode
    x_axis_mode = get_x_axis_mode()
    if x_axis_mode == 'frame':
        x_data = frames
        x_label = 'Frame Number'
    else:
        x_data = time
        x_label = 'Time (seconds)'
    
    # Get zoom parameters
    zoom_start, zoom_end = get_zoom_frames(fps, len(df))
    
    if x_axis_mode == 'frame':
        zoom_x_start = zoom_start
        zoom_x_end = zoom_end
    else:
        zoom_x_start = zoom_start / fps
        zoom_x_end = zoom_end / fps
    
    # Determine y-axis label
    if use_3d:
        y_label = 'Displacement (distance change)'
    else:
        if resolution is not None:
            y_label = 'Displacement (pixels per frame)'
        else:
            y_label = 'Displacement (distance change)'
    
    for point_a, point_b in point_pairs:
        try:
            distances = np.array(calculate_distance_for_pair(df, point_a, point_b, 
                                                             CONFIG['POINTS_PREFIX'], use_3d, resolution))
            
            # Calculate frame-to-frame change (displacement/velocity)
            displacements = np.diff(distances)
            displacements = np.insert(displacements, 0, 0)  # Add 0 for first frame
            
            label = f"Point {point_a} ↔ Point {point_b}"
            
            # Plot full view
            ax1.plot(x_data, displacements, marker='o', markersize=2, label=label, alpha=0.7)
            
            # Plot zoomed view
            ax2.plot(x_data, displacements, marker='o', markersize=3, label=label, alpha=0.7)
            
        except Exception as e:
            print(f"Error analyzing pair ({point_a}, {point_b}): {e}")
    
    # Configure full view subplot
    ax1.set_xlabel(x_label, fontsize=11)
    ax1.set_ylabel(y_label, fontsize=11)
    ax1.set_title('Frame-to-Frame Distance Changes (Displacement) - Full View', fontsize=13)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Highlight zoom region on full view
    ax1.axvspan(zoom_x_start, zoom_x_end, alpha=0.2, color='yellow', 
                label=f'Zoom region')
    
    # Configure zoomed view subplot
    ax2.set_xlim(zoom_x_start, zoom_x_end)
    ax2.set_xlabel(x_label, fontsize=11)
    ax2.set_ylabel(y_label, fontsize=11)
    
    if x_axis_mode == 'frame':
        ax2.set_title(f"Zoomed View: Frames {zoom_start}-{zoom_end}", fontsize=13)
    else:
        ax2.set_title(f"Zoomed View: Frames {zoom_start}-{zoom_end} ({zoom_x_start:.2f}s - {zoom_x_end:.2f}s)", 
                      fontsize=13)
    
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to analyze and plot keypoint distances and displacements."""
    print("=" * 60)
    print("Facial Keypoint Distance & Displacement Analysis")
    print("=" * 60)

    output_dir = "/".join(CONFIG['CSV_PATH'].split('/')[:-1])
    print(f"Output directory: {output_dir}")
    
    # Load data
    df = load_keypoints(CONFIG['CSV_PATH'])
    
    # Get resolution from config
    resolution = CONFIG.get('RESOLUTION')
    if resolution:
        print(f"Video resolution: {resolution[0]}x{resolution[1]} pixels")
    
    # Validate point pairs
    points_prefix = CONFIG['POINTS_PREFIX']
    for point_a, point_b in CONFIG['POINT_PAIRS']:
        x_col_a = f'{points_prefix}_{point_a}_x'
        x_col_b = f'{points_prefix}_{point_b}_x'
        if x_col_a not in df.columns:
            print(f"Warning: Point {point_a} not found in CSV")
        if x_col_b not in df.columns:
            print(f"Warning: Point {point_b} not found in CSV")
    
    zoom_mode = CONFIG.get('ZOOM_MODE', 'frame')
    x_axis_mode = get_x_axis_mode()
    
    if zoom_mode == 'time':
        time_config = CONFIG.get('TIME_MODE', {})
        zoom_duration = time_config.get('ZOOM_DURATION', [0.0, 1.0])
        print(f"\nAnalyzing {len(CONFIG['POINT_PAIRS'])} point pairs")
        print(f"FPS: {CONFIG['FPS']}")
        print(f"Total duration: {len(df) / CONFIG['FPS']:.2f} seconds")
        print(f"Zoom mode: time")
        print(f"X-axis mode: {x_axis_mode}")
        print(f"Zoom duration: {zoom_duration[0]:.2f}s - {zoom_duration[1]:.2f}s")
        print(f"               (Frames {int(zoom_duration[0]*CONFIG['FPS'])}-{int(zoom_duration[1]*CONFIG['FPS'])})")
    else:
        frame_config = CONFIG.get('FRAME_MODE', {})
        zoom_duration = frame_config.get('ZOOM_DURATION', [0, len(df)])
        print(f"\nAnalyzing {len(CONFIG['POINT_PAIRS'])} point pairs")
        print(f"FPS: {CONFIG['FPS']}")
        print(f"Total duration: {len(df) / CONFIG['FPS']:.2f} seconds")
        print(f"Zoom mode: frame")
        print(f"X-axis mode: {x_axis_mode}")
        print(f"Zoom duration: Frames {zoom_duration[0]}-{zoom_duration[1]}")
        print(f"               ({zoom_duration[0]/CONFIG['FPS']:.2f}s - {zoom_duration[1]/CONFIG['FPS']:.2f}s)")
    
    # Plot absolute distances (3D)
    # print("\n" + "=" * 60)
    # print("3D Distance Analysis")
    # print("=" * 60)
    # plot_distances(df, points_prefix, CONFIG['POINT_PAIRS'], CONFIG['FPS'], 
    #                use_3d=True, output_dir=output_dir)
    
    # Plot 2D distances in pixels
    print("\n" + "=" * 60)
    print("2D Distance Analysis (in pixels)")
    print("=" * 60)
    plot_distances(df, points_prefix, CONFIG['POINT_PAIRS'], CONFIG['FPS'], 
                   use_3d=False, output_dir=output_dir, resolution=resolution)
    
    # Optional: Analyze displacement (distance changes)
    # print("\n" + "=" * 60)
    # print("Displacement Analysis (in pixels)")
    # print("=" * 60)
    # analyze_displacement_changes(df, CONFIG['POINT_PAIRS'], CONFIG['FPS'], 
    #                              use_3d=False, resolution=resolution)



if __name__ == "__main__":
    main()