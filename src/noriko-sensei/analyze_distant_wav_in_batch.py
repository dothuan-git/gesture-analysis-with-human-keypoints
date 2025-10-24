import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import re
import wave


# Configuration constants
CONFIG = {
    'INPUT_FOLDER': 'workspaces/IMG_0004/chunks_006/landmarks',
    'AUDIO_FOLDER': 'workspaces/IMG_0004/chunks_006/audio',
    'OUTPUT_SUBFOLDER': 'plots',
    'CSV_PATTERN': '*_face.csv',
    'AUDIO_EXTENSION': '.wav',
    'FPS': 30, 
    'POINT_PAIRS': [
        # Define pairs of landmark indices to calculate distances
        (0, 17, 'Point 0-17'),
        # Add more pairs as needed
    ],
    'PLOT_DPI': 100,
    'PLOT_FIGSIZE': (12, 8),
    'SAVE_INDIVIDUAL_PLOTS': True,  # Save separate plot for each video
}


def parse_csv_header(header: List[str]) -> Dict[int, List[int]]:
    """
    Parse CSV header to extract landmark indices and their column positions.
    Returns: {landmark_index: [x_col, y_col, z_col]}
    """
    landmark_map = {}
    
    for col_idx, col_name in enumerate(header):
        # Match pattern like 'face_37_x', 'face_37_y', 'face_37_z'
        match = re.match(r'face_(\d+)_(x|y|z)', col_name)
        if match:
            landmark_idx = int(match.group(1))
            coord_type = match.group(2)
            
            if landmark_idx not in landmark_map:
                landmark_map[landmark_idx] = [None, None, None]
            
            coord_idx = {'x': 0, 'y': 1, 'z': 2}[coord_type]
            landmark_map[landmark_idx][coord_idx] = col_idx
    
    return landmark_map


def load_audio_waveform(audio_path: str) -> Optional[Tuple[np.ndarray, int]]:
    """
    Load audio file and return waveform data and sample rate.
    Returns: (audio_data, sample_rate) or None if error
    """
    try:
        with wave.open(audio_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            audio_data = wav_file.readframes(n_frames)
            
            # Convert bytes to numpy array
            if wav_file.getsampwidth() == 1:
                dtype = np.uint8
            elif wav_file.getsampwidth() == 2:
                dtype = np.int16
            else:
                dtype = np.int32
            
            audio_array = np.frombuffer(audio_data, dtype=dtype)
            
            # Convert to mono if stereo
            if wav_file.getnchannels() == 2:
                audio_array = audio_array.reshape(-1, 2).mean(axis=1)
            
            return audio_array, sample_rate
    except Exception as e:
        print(f"  Warning: Could not load audio {audio_path}: {e}")
        return None


def calculate_2d_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two 2D points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def load_and_calculate_distances(csv_path: str, point_pairs: List[Tuple[int, int, str]]) -> Dict[str, List[float]]:
    """
    Load CSV file and calculate distances for specified point pairs.
    Returns: {pair_label: [distances for each frame]}
    """
    print(f"Processing: {Path(csv_path).name}")
    
    distances = {label: [] for _, _, label in point_pairs}
    frames = []
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            # Parse header to get landmark column positions
            landmark_map = parse_csv_header(header)
            
            # Check if all required landmarks exist
            required_landmarks = set()
            for p1, p2, _ in point_pairs:
                required_landmarks.add(p1)
                required_landmarks.add(p2)
            
            missing_landmarks = required_landmarks - set(landmark_map.keys())
            if missing_landmarks:
                print(f"  Warning: Missing landmarks {missing_landmarks} in {Path(csv_path).name}")
                return None
            
            # Process each frame
            for row in reader:
                try:
                    frame_num = int(row[0])
                    frames.append(frame_num)
                    
                    # Calculate distance for each point pair
                    for p1_idx, p2_idx, label in point_pairs:
                        # Get x, y coordinates for both points
                        x1_col, y1_col, _ = landmark_map[p1_idx]
                        x2_col, y2_col, _ = landmark_map[p2_idx]
                        
                        x1 = float(row[x1_col])
                        y1 = float(row[y1_col])
                        x2 = float(row[x2_col])
                        y2 = float(row[y2_col])
                        
                        dist = calculate_2d_distance(x1, y1, x2, y2)
                        distances[label].append(dist)
                        
                except (ValueError, IndexError) as e:
                    print(f"  Warning: Error processing row {len(frames)}: {e}")
                    continue
        
        print(f"  Processed {len(frames)} frames")
        return {'frames': frames, 'distances': distances}
        
    except Exception as e:
        print(f"  Error loading {csv_path}: {e}")
        return None


def plot_distances_individual(data: Dict, video_name: str, fps: float, output_dir: str, 
                              point_pairs: List[Tuple[int, int, str]], audio_folder: str):
    """Create and save individual plot for one video with audio waveform."""
    # Try to load audio file
    audio_path = os.path.join(audio_folder, f'{video_name}{CONFIG["AUDIO_EXTENSION"]}')
    audio_data = None
    sample_rate = None
    
    if os.path.exists(audio_path):
        result = load_audio_waveform(audio_path)
        if result:
            audio_data, sample_rate = result
            print(f"  Loaded audio: {Path(audio_path).name}")
    else:
        print(f"  Warning: Audio file not found: {Path(audio_path).name}")
    
    # Create subplots: distance plots + audio waveform
    n_plots = len(point_pairs) + (1 if audio_data is not None else 0)
    fig, axes = plt.subplots(n_plots, 1, figsize=CONFIG['PLOT_FIGSIZE'], 
                             squeeze=False)
    
    # Always flatten to ensure we have a list
    axes = axes.flatten()
    
    frames = data['frames']
    distances = data['distances']
    
    # Plot distances
    for idx, (p1, p2, label) in enumerate(point_pairs):
        ax = axes[idx]
        # Compute time axis from FPS or audio duration
        if audio_data is not None:
            total_audio_time = len(audio_data) / sample_rate
            time_axis = np.linspace(0, total_audio_time, len(frames))
        else:
            time_axis = np.array(frames) / fps

        ax.plot(time_axis, distances[label], linewidth=1, marker='o', markersize=3)
        ax.set_xlabel('Time (seconds)')

        ax.set_ylabel('Distance (normalized)')
        ax.set_title(f'{video_name} - {label} (Points {p1}-{p2})')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_dist = np.mean(distances[label])
        std_dist = np.std(distances[label])
        ax.axhline(mean_dist, color='red', linestyle='--', linewidth=1, 
                   label=f'Mean: {mean_dist:.4f}')
        ax.legend(loc='upper right')
    
    # Plot audio waveform if available
    if audio_data is not None:
        ax = axes[-1]
        time_axis = np.arange(len(audio_data)) / sample_rate
        ax.plot(time_axis, audio_data, linewidth=0.5, color='forestgreen')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'{video_name} - Audio Waveform')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'{video_name}_distances.png')
    plt.savefig(output_path, dpi=CONFIG['PLOT_DPI'], bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_distances_combined(all_data: Dict[str, Dict], output_dir: str, point_pairs: List[Tuple[int, int, str]]):
    """Create and save combined plot with all videos."""
    # This function is removed - no combined plots needed
    pass


def save_distance_summary(all_data: Dict[str, Dict], output_dir: str, point_pairs: List[Tuple[int, int, str]]):
    """Save summary statistics to CSV."""
    summary_path = os.path.join(output_dir, 'distance_summary.csv')
    
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['video', 'point_pair', 'mean', 'std', 'min', 'max', 'median']
        writer.writerow(header)
        
        # Data for each video and point pair
        for video_name, data in all_data.items():
            for p1, p2, label in point_pairs:
                distances = data['distances'][label]
                
                row = [
                    video_name,
                    f'{p1}-{p2} ({label})',
                    f'{np.mean(distances):.6f}',
                    f'{np.std(distances):.6f}',
                    f'{np.min(distances):.6f}',
                    f'{np.max(distances):.6f}',
                    f'{np.median(distances):.6f}'
                ]
                writer.writerow(row)
    
    print(f"Saved summary: {summary_path}")


def main():
    """Main function to process landmark CSVs and generate distance plots."""
    input_folder = CONFIG['INPUT_FOLDER']
    audio_folder = CONFIG['AUDIO_FOLDER']
    csv_pattern = CONFIG['CSV_PATTERN']
    fps = CONFIG['FPS']
    
    if not os.path.exists(input_folder):
        print(f"Error: Input folder not found: {input_folder}")
        return
    
    if not os.path.exists(audio_folder):
        print(f"Warning: Audio folder not found: {audio_folder}")
        print("Continuing without audio waveforms...")
    
    # Create output directory
    parent_dir = str(Path(input_folder).parent)
    output_dir = os.path.join(parent_dir, CONFIG['OUTPUT_SUBFOLDER'])
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Input folder: {input_folder}")
    print(f"Audio folder: {audio_folder}")
    print(f"Output folder: {output_dir}")
    print(f"\nPoint pairs to analyze:")
    for p1, p2, label in CONFIG['POINT_PAIRS']:
        print(f"  - Points {p1}-{p2}: {label}")
    
    # Find all CSV files
    csv_files = sorted(glob.glob(os.path.join(input_folder, csv_pattern)))
    
    if not csv_files:
        print(f"\nNo CSV files found matching pattern: {csv_pattern}")
        return
    
    print(f"\nFound {len(csv_files)} CSV file(s):")
    for csv_file in csv_files:
        print(f"  - {Path(csv_file).name}")
    
    # Process each CSV file
    all_data = {}
    
    print("\n" + "="*60)
    print("Processing CSV files...")
    print("="*60)
    
    for csv_path in csv_files:
        video_name = Path(csv_path).stem.replace('_face', '')
        
        # Load and calculate distances
        result = load_and_calculate_distances(csv_path, CONFIG['POINT_PAIRS'])
        
        if result is None:
            print(f"  Skipping {video_name} due to errors")
            continue
        
        all_data[video_name] = result
        
        # Create individual plot if enabled
        if CONFIG['SAVE_INDIVIDUAL_PLOTS']:
            plot_distances_individual(result, video_name, fps, output_dir, CONFIG['POINT_PAIRS'], audio_folder)
    
    if not all_data:
        print("\nNo data processed successfully")
        return
    
    # Save summary statistics
    print("\n" + "="*60)
    print("Saving summary statistics...")
    print("="*60)
    save_distance_summary(all_data, output_dir, CONFIG['POINT_PAIRS'])
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Videos processed: {len(all_data)}")
    print(f"Point pairs analyzed: {len(CONFIG['POINT_PAIRS'])}")
    print(f"Output saved to: {output_dir}")
    print("\nGenerated files:")
    if CONFIG['SAVE_INDIVIDUAL_PLOTS']:
        print(f"  - {len(all_data)} individual plot(s) with audio waveforms")
    print(f"  - 1 summary CSV")


if __name__ == "__main__":
    main()