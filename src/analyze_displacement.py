import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Configuration constants
CONFIG = {
    'CSV_PATH': 'data/IMG_0004/face.csv',
    'FPS': 30.0,  # Frames per second of the video
    'POINT_PAIRS': [
        (0, 17),     # Mouth corners
    ],
    'ZOOM_DURATION': [12000, 14000],  # [start_frame, end_frame] for zoomed subplot
    'X_AXIS_MODE': 'frame',  # 'time' (seconds) or 'frame' (frame numbers)
    'PLOT_TITLE': 'Facial Keypoint Displacement Over Time',
    'FIGURE_SIZE': (24, 12),
    'SAVE_PLOT': True,
    'OUTPUT_PATH': 'displacement_plot.png',
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


def extract_point_coords(df: pd.DataFrame, point_id: int, frame_idx: int) -> Tuple[float, float, float]:
    """Extract x, y, z coordinates for a specific point and frame."""
    x_col = f'face_{point_id}_x'
    y_col = f'face_{point_id}_y'
    z_col = f'face_{point_id}_z'
    
    if x_col not in df.columns or y_col not in df.columns or z_col not in df.columns:
        raise ValueError(f"Point {point_id} not found in CSV columns")
    
    x = df.loc[frame_idx, x_col]
    y = df.loc[frame_idx, y_col]
    z = df.loc[frame_idx, z_col]
    
    return x, y, z


def calculate_displacement_for_pair(df: pd.DataFrame, point_a: int, point_b: int, 
                                    use_3d: bool = True) -> List[float]:
    """Calculate displacement between two points for all frames."""
    displacements = []
    
    for frame_idx in range(len(df)):
        try:
            x1, y1, z1 = extract_point_coords(df, point_a, frame_idx)
            x2, y2, z2 = extract_point_coords(df, point_b, frame_idx)
            
            if use_3d:
                distance = calculate_euclidean_distance(x1, y1, z1, x2, y2, z2)
            else:
                distance = calculate_2d_distance(x1, y1, x2, y2)
            
            displacements.append(distance)
        except Exception as e:
            print(f"Warning: Error at frame {frame_idx} for pair ({point_a}, {point_b}): {e}")
            displacements.append(np.nan)
    
    return displacements


def frames_to_time(frames: np.ndarray, fps: float) -> np.ndarray:
    """Convert frame numbers to time in seconds."""
    return frames / fps


def plot_displacements(df: pd.DataFrame, point_pairs: List[Tuple[int, int]], 
                      fps: float, use_3d: bool = True):
    """Plot displacement for all point pairs over time with zoomed subplot."""
    fig = plt.figure(figsize=CONFIG['FIGURE_SIZE'])
    
    # Create two subplots: full view and zoomed view
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    
    frames = np.arange(len(df))
    time = frames_to_time(frames, fps)
    
    # Determine x-axis data based on mode
    x_axis_mode = CONFIG.get('X_AXIS_MODE', 'time').lower()
    if x_axis_mode == 'frame':
        x_data = frames
        x_label = 'Frame Number'
    else:
        x_data = time
        x_label = 'Time (seconds)'
    
    # Get zoom parameters
    zoom_start, zoom_end = CONFIG.get('ZOOM_DURATION', [0, len(df)])
    zoom_start = max(0, min(zoom_start, len(df) - 1))
    zoom_end = max(zoom_start + 1, min(zoom_end, len(df)))
    
    if x_axis_mode == 'frame':
        zoom_x_start = zoom_start
        zoom_x_end = zoom_end
    else:
        zoom_x_start = zoom_start / fps
        zoom_x_end = zoom_end / fps
    
    for point_a, point_b in point_pairs:
        try:
            displacements = calculate_displacement_for_pair(df, point_a, point_b, use_3d)
            label = f"Point {point_a} ↔ Point {point_b}"
            
            # Plot full view
            ax1.plot(x_data, displacements, marker='o', markersize=2, label=label, alpha=0.7)
            
            # Plot zoomed view
            ax2.plot(x_data, displacements, marker='o', markersize=3, label=label, alpha=0.7)
            
            # Print statistics
            mean_disp = np.nanmean(displacements)
            std_disp = np.nanstd(displacements)
            print(f"\n{label}:")
            print(f"  Mean displacement: {mean_disp:.6f}")
            print(f"  Std deviation: {std_disp:.6f}")
            print(f"  Min: {np.nanmin(displacements):.6f}")
            print(f"  Max: {np.nanmax(displacements):.6f}")
            
        except Exception as e:
            print(f"Error plotting pair ({point_a}, {point_b}): {e}")
    
    # Configure full view subplot
    dimension = "3D" if use_3d else "2D"
    ax1.set_xlabel(x_label, fontsize=11)
    ax1.set_ylabel('Displacement (normalized units)', fontsize=11)
    ax1.set_title(f"{CONFIG['PLOT_TITLE']} ({dimension}) - Full View", fontsize=13)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Highlight zoom region on full view
    ax1.axvspan(zoom_x_start, zoom_x_end, alpha=0.2, color='yellow', 
                label=f'Zoom region')
    
    # Configure zoomed view subplot
    ax2.set_xlim(zoom_x_start, zoom_x_end)
    ax2.set_xlabel(x_label, fontsize=11)
    ax2.set_ylabel('Displacement (normalized units)', fontsize=11)
    
    if x_axis_mode == 'frame':
        ax2.set_title(f"Zoomed View: Frames {zoom_start}-{zoom_end}", fontsize=13)
    else:
        ax2.set_title(f"Zoomed View: Frames {zoom_start}-{zoom_end} ({zoom_x_start:.2f}s - {zoom_x_end:.2f}s)", 
                      fontsize=13)
    
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if CONFIG['SAVE_PLOT']:
        plt.savefig(CONFIG['OUTPUT_PATH'], dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {CONFIG['OUTPUT_PATH']}")
    
    plt.show()


def analyze_displacement_changes(df: pd.DataFrame, point_pairs: List[Tuple[int, int]], 
                                fps: float, use_3d: bool = True):
    """Analyze frame-to-frame displacement changes (velocity) with zoomed subplot."""
    fig = plt.figure(figsize=CONFIG['FIGURE_SIZE'])
    
    # Create two subplots: full view and zoomed view
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    
    frames = np.arange(len(df))
    time = frames_to_time(frames, fps)
    
    # Determine x-axis data based on mode
    x_axis_mode = CONFIG.get('X_AXIS_MODE', 'time').lower()
    if x_axis_mode == 'frame':
        x_data = frames
        x_label = 'Frame Number'
    else:
        x_data = time
        x_label = 'Time (seconds)'
    
    # Get zoom parameters
    zoom_start, zoom_end = CONFIG.get('ZOOM_DURATION', [0, len(df)])
    zoom_start = max(0, min(zoom_start, len(df) - 1))
    zoom_end = max(zoom_start + 1, min(zoom_end, len(df)))
    
    if x_axis_mode == 'frame':
        zoom_x_start = zoom_start
        zoom_x_end = zoom_end
    else:
        zoom_x_start = zoom_start / fps
        zoom_x_end = zoom_end / fps
    
    for point_a, point_b in point_pairs:
        try:
            displacements = np.array(calculate_displacement_for_pair(df, point_a, point_b, use_3d))
            
            # Calculate frame-to-frame change (first derivative)
            changes = np.diff(displacements)
            changes = np.insert(changes, 0, 0)  # Add 0 for first frame
            
            label = f"Point {point_a} ↔ Point {point_b}"
            
            # Plot full view
            ax1.plot(x_data, changes, marker='o', markersize=2, label=label, alpha=0.7)
            
            # Plot zoomed view
            ax2.plot(x_data, changes, marker='o', markersize=3, label=label, alpha=0.7)
            
        except Exception as e:
            print(f"Error analyzing pair ({point_a}, {point_b}): {e}")
    
    # Configure full view subplot
    ax1.set_xlabel(x_label, fontsize=11)
    ax1.set_ylabel('Displacement Change (velocity)', fontsize=11)
    ax1.set_title('Frame-to-Frame Displacement Changes - Full View', fontsize=13)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Highlight zoom region on full view
    ax1.axvspan(zoom_x_start, zoom_x_end, alpha=0.2, color='yellow', 
                label=f'Zoom region')
    
    # Configure zoomed view subplot
    ax2.set_xlim(zoom_x_start, zoom_x_end)
    ax2.set_xlabel(x_label, fontsize=11)
    ax2.set_ylabel('Displacement Change (velocity)', fontsize=11)
    
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
    """Main function to analyze and plot keypoint displacements."""
    print("=" * 60)
    print("Facial Keypoint Displacement Analysis")
    print("=" * 60)
    
    # Load data
    df = load_keypoints(CONFIG['CSV_PATH'])
    
    # Validate point pairs
    for point_a, point_b in CONFIG['POINT_PAIRS']:
        x_col_a = f'face_{point_a}_x'
        x_col_b = f'face_{point_b}_x'
        if x_col_a not in df.columns:
            print(f"Warning: Point {point_a} not found in CSV")
        if x_col_b not in df.columns:
            print(f"Warning: Point {point_b} not found in CSV")
    
    print(f"\nAnalyzing {len(CONFIG['POINT_PAIRS'])} point pairs")
    print(f"FPS: {CONFIG['FPS']}")
    print(f"Total duration: {len(df) / CONFIG['FPS']:.2f} seconds")
    print(f"X-axis mode: {CONFIG.get('X_AXIS_MODE', 'time')}")
    print(f"Zoom duration: Frames {CONFIG['ZOOM_DURATION'][0]}-{CONFIG['ZOOM_DURATION'][1]}")
    print(f"               ({CONFIG['ZOOM_DURATION'][0]/CONFIG['FPS']:.2f}s - {CONFIG['ZOOM_DURATION'][1]/CONFIG['FPS']:.2f}s)")
    
    # Plot absolute displacements (3D)
    # print("\n" + "=" * 60)
    # print("3D Displacement Analysis")
    # print("=" * 60)
    # plot_displacements(df, CONFIG['POINT_PAIRS'], CONFIG['FPS'], use_3d=True)
    
    # Optional: Plot 2D displacements (x, y only)
    print("\n" + "=" * 60)
    print("2D Displacement Analysis")
    print("=" * 60)
    plot_displacements(df, CONFIG['POINT_PAIRS'], CONFIG['FPS'], use_3d=False)
    
    # Optional: Analyze displacement changes (velocity)
    print("\n" + "=" * 60)
    print("Displacement Change Analysis")
    print("=" * 60)
    analyze_displacement_changes(df, CONFIG['POINT_PAIRS'], CONFIG['FPS'], use_3d=True)


if __name__ == "__main__":
    main()