import pandas as pd
import cv2
import os
from datetime import datetime

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils import *

CONFIG = {
    'VIDEO_PATH': 'assets/IMG_0004.MP4',
    'CSV_PATH': 'data/FURUKAWA-PilotExp.xlsx',
    'OUTPUT_DIR': '',
    'PADDING_SECONDS': 1,  # Number of seconds to pad before and after each segment
}


def read_file(file_path):
    """
    Read CSV or XLSX file based on file extension.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.csv':
        return pd.read_csv(file_path)
    elif file_extension in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Only .csv, .xlsx, and .xls are supported.")
    

def time_to_seconds(time_str):
    """
    Convert time string in format 'hour:minute:second' to total seconds.
    Example: '1:10:18.00' -> 4218.0
    """
    if pd.isna(time_str) or time_str == '':
        return None
    
    try:
        # Handle format like '1:10:18.00' or '0:10:18'
        parts = str(time_str).split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    except Exception as e:
        print(f"Error parsing time '{time_str}': {e}")
        return None


def cut_video_segment(video_path, start_time, end_time, output_path, fps, padding_seconds=0):
    """
    Cut a segment from video and save it with frame numbers overlaid.
    
    Args:
        video_path: Path to input video
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Path to save output video
        fps: Frames per second of the video
        padding_seconds: Number of seconds to pad before and after the segment
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create output writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Apply padding and ensure we don't go out of bounds
    padded_start_time = max(0, start_time - padding_seconds)
    padded_end_time = min(end_time + padding_seconds, total_frames / fps)
    
    # Calculate frame positions
    start_frame = int(padded_start_time * fps)
    end_frame = int(padded_end_time * fps)
    
    # Set video to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add frame number and timestamp overlay
        text1 = f"frame: {current_frame}"
        text2 = f"time: {current_frame/fps:.2f}s"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        color = (0, 255, 0)  # Green color

        # Get text size for both lines
        (text1_width, text1_height), baseline1 = cv2.getTextSize(text1, font, font_scale, thickness)
        (text2_width, text2_height), baseline2 = cv2.getTextSize(text2, font, font_scale, thickness)

        # Position for top-right corner
        text1_x = width - text1_width - 10
        text1_y = 40
        line_spacing = 20  # Space between lines
        text2_x = width - text2_width - 10
        text2_y = text1_y + text1_height + line_spacing

        # Draw semi-transparent background for first line
        overlay = frame.copy()
        cv2.rectangle(overlay, (text1_x - 5, text1_y - text1_height - 5), 
                    (text1_x + text1_width + 5, text1_y + baseline1 + 5), 
                    (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(frame, text1, (text1_x, text1_y), font, font_scale, color, thickness)

        # Draw semi-transparent background for second line
        overlay = frame.copy()
        cv2.rectangle(overlay, (text2_x - 5, text2_y - text2_height - 5), 
                    (text2_x + text2_width + 5, text2_y + baseline2 + 5), 
                    (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(frame, text2, (text2_x, text2_y), font, font_scale, color, thickness)
                
        out.write(frame)
        current_frame += 1
    
    cap.release()
    out.release()
    
    return start_frame, end_frame


def main():
    # Create output directory
    output_dir = CONFIG["OUTPUT_DIR"]
    video_name = CONFIG["VIDEO_PATH"].split("/")[-1].split(".")[0]
    if not os.path.exists(output_dir) or len(output_dir) == 0:
        root_path = f'workspaces/{video_name}'
        output_dir = create_incremented_dir(root_path, 'chunks')
    
    # Create log file for all print statements
    log_file_path = os.path.join(output_dir, 'processing_output.txt')
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    def emit(message):
        """Print to console and write to log file"""
        print(message)
        log_file.write(message + '\n')
        log_file.flush()
    
    # Read CSV or XLSX file
    emit(f"Reading file: {CONFIG['CSV_PATH']}")
    df_original = read_file(CONFIG['CSV_PATH'])
    
    # Store original column names before lowercasing
    original_columns = df_original.columns.tolist()
    
    # Lower case column names for processing
    df_original.columns = df_original.columns.str.lower()
    
    # Extract required columns and keep track of original indices
    df_segments = df_original[['start', 'end', 'category']].copy()
    df_segments['original_index'] = df_original.index
    
    # Convert time strings to seconds
    df_segments['start_seconds'] = df_segments['start'].apply(time_to_seconds)
    df_segments['end_seconds'] = df_segments['end'].apply(time_to_seconds)
    
    # Remove rows with empty start or end times
    df_segments = df_segments.dropna(subset=['start_seconds', 'end_seconds'])
    
    # Sort by start time and create new index for segment IDs
    df_segments = df_segments.sort_values('start_seconds').reset_index(drop=True)
    df_segments['segment_id'] = df_segments.index + 1
    
    # Open video to get properties
    cap = cv2.VideoCapture(CONFIG['VIDEO_PATH'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    
    padding_seconds = CONFIG.get('PADDING_SECONDS', 0)
    
    emit(f"Video FPS: {fps}")
    emit(f"Padding: {padding_seconds} seconds before and after each segment")
    emit(f"\nProcessing {len(df_segments)} segments...")
    
    # Initialize log data
    processing_log_data = []
    segments_log_data = []
    
    # Process each segment
    for idx, row in df_segments.iterrows():
        segment_id = row['segment_id']
        segment_name = f"{video_name}_segment{segment_id}"
        category = row['category']
        start_seconds = row['start_seconds']
        end_seconds = row['end_seconds']
        duration = end_seconds - start_seconds
        original_start = row['start']
        original_end = row['end']
        
        # Create output filename
        output_filename = f"{segment_name}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        emit(f"- {segment_name}: {category} [{start_seconds}s - {end_seconds}s] - duration: {duration:.2f}s")
        if padding_seconds > 0:
            emit(f"  (with padding {padding_seconds}s: {max(0, start_seconds - padding_seconds)}s - {min(end_seconds + padding_seconds, max_frame/fps)}s)")

        # Cut video segment with padding
        start_frame, end_frame = cut_video_segment(
            CONFIG['VIDEO_PATH'],
            start_seconds,
            end_seconds,
            output_path,
            fps,
            padding_seconds=padding_seconds
        )
        
        # Get all original columns for this row (using original index)
        original_row_index = row['original_index']
        original_row_data = df_original.loc[original_row_index].to_dict()
        
        # Create processing log entry with segment_name, category, original times, duration, and all original columns
        processing_log_entry = {
            'segment_name': segment_name,
            'category': category,
            'original_start': original_start,
            'original_end': original_end,
            'duration (s)': duration
        }
        # Add all other original columns
        for col in df_original.columns:
            if col not in ['start', 'end', 'category']:
                processing_log_entry[col] = original_row_data[col]
        
        processing_log_data.append(processing_log_entry)
        
        # Create segments log entry
        segments_log_data.append({
            'segment_name': segment_name,
            'category': category,
            'original_start': original_start,
            'original_end': original_end,
            'start (s)': start_seconds,
            'end (s)': end_seconds,
            'duration (s)': duration,
            'padding (s)': padding_seconds,
            'start (frame)': start_frame,
            'end (frame)': end_frame
        })
        
        emit(f"  Saved: {output_filename}")
    
    # Save processing log CSV (with all original columns)
    log_csv_path = os.path.join(output_dir, 'processing_log.csv')
    log_df = pd.DataFrame(processing_log_data)
    log_df.to_csv(log_csv_path, index=False)
    emit(f"\nProcessing log saved to: {log_csv_path}")

    # Save segments log CSV
    segments_csv_path = os.path.join(output_dir, 'segments_log.csv')   
    segments_df = pd.DataFrame(segments_log_data)
    segments_df.to_csv(segments_csv_path, index=False)
    emit(f"Segments log saved to: {segments_csv_path}")
    emit(f"Total segments processed: {len(processing_log_data)}")
    emit(f"\nAll output saved to log file: {log_file_path}")
    
    # Close log file
    log_file.close()


if __name__ == "__main__":
    main()