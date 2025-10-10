import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Configuration constants
CONFIG = {
    'CSV_PATH': 'data/IMG_0004/face.csv',
    'FPS': 30.0,  # Frames per second of the video
    'POINT_PAIRS': [
        (0, 17),
    ],
    'PLOT_TITLE': 'Facial Keypoint Displacement Over Time',
    'FIGURE_SIZE': (12, 8),
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
    """Plot displacement for all point pairs over time."""
    fig, ax = plt.subplots(figsize=CONFIG['FIGURE_SIZE'])
    
    frames = np.arange(len(df))
    time = frames_to_time(frames, fps)
    
    for point_a, point_b in point_pairs:
        try:
            displacements = calculate_displacement_for_pair(df, point_a, point_b, use_3d)
            label = f"Point {point_a} ↔ Point {point_b}"
            ax.plot(time, displacements, marker='o', markersize=2, label=label, alpha=0.7)
            
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
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Displacement (normalized units)', fontsize=12)
    dimension = "3D" if use_3d else "2D"
    ax.set_title(f"{CONFIG['PLOT_TITLE']} ({dimension})", fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if CONFIG['SAVE_PLOT']:
        plt.savefig(CONFIG['OUTPUT_PATH'], dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {CONFIG['OUTPUT_PATH']}")
    
    plt.show()


def analyze_displacement_changes(df: pd.DataFrame, point_pairs: List[Tuple[int, int]], 
                                fps: float, use_3d: bool = True):
    """Analyze frame-to-frame displacement changes (velocity)."""
    fig, ax = plt.subplots(figsize=CONFIG['FIGURE_SIZE'])
    
    frames = np.arange(len(df))
    time = frames_to_time(frames, fps)
    
    for point_a, point_b in point_pairs:
        try:
            displacements = np.array(calculate_displacement_for_pair(df, point_a, point_b, use_3d))
            
            # Calculate frame-to-frame change (first derivative)
            changes = np.diff(displacements)
            changes = np.insert(changes, 0, 0)  # Add 0 for first frame
            
            label = f"Point {point_a} ↔ Point {point_b}"
            ax.plot(time, changes, marker='o', markersize=2, label=label, alpha=0.7)
            
        except Exception as e:
            print(f"Error analyzing pair ({point_a}, {point_b}): {e}")
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Displacement Change (velocity)', fontsize=12)
    ax.set_title('Frame-to-Frame Displacement Changes', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
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