# Noriko-Sensei Module Documentation

## Overview

The **noriko-sensei** module is a batch processing extension for analyzing video and audio (waveforms) in parallel. It's designed for processing multiple video segments with synchronized audio data to extract and analyze human gesture keypoints at scale.

This module is used for analyzing structured datasets where you have predefined temporal segments (start/end times) defined in a CSV or XLSX file.

## Prerequisites

### Required Installation

Before using the noriko-sensei module, ensure you have **FFmpeg** installed on your system:

- **Windows**: Download from https://ffmpeg.org/download.html or install via package manager
- **macOS**: `brew install ffmpeg`
- **Linux**: `apt install ffmpeg`

Verify installation:
```bash
ffmpeg -version
```

### Dependencies

All Python dependencies are listed in `requirements.txt`. Install via:
```bash
pip install -r requirements.txt
```

Key packages:
- **MediaPipe 0.10.21+**: For landmark detection
- **OpenCV 4.12.0+**: For video processing
- **Pandas 2.3.3+**: For data handling
- **NumPy 2.2.6+**: For numerical operations
- **Matplotlib 3.10.7+**: For plotting

## Workflow Overview

The noriko-sensei module provides a complete pipeline for batch processing video and audio:

```
CSV/XLSX Segments File
         ↓
[1] video_cutting.py (Required)
    ├─ Input: Video file + segments CSV/XLSX
    ├─ Output: Cut video segments + extracted audio
    └─ Creates: chunks_NNN/ workspace
         ↓
[2] extract_landmarks_in_batch.py (Required)
    ├─ Input: Cut video segments (from step 1)
    ├─ Output: Landmark CSVs (face, hands)
    └─ Creates: landmarks/ subdirectory
         ↓
[3] analyze_distant_wav_in_batch.py (Required)
    ├─ Input: Landmark CSVs + Audio files
    ├─ Output: Distance plots + summary statistics
    └─ Creates: plots/ subdirectory
         ↓
[Optional] Demo Scripts (Real-time Analysis)
    ├─ face_points_distant_wav_demo.py
    └─ face_points_distant_wav_demo2.py
       (Visualize analysis in real-time with audio overlay)
```

## Input Format: CSV/XLSX Segments File

The noriko-sensei workflow starts with a **segments file** (CSV or XLSX) that defines temporal segments to extract from a video.

### Required Columns (Obligated 3 columns)

Your CSV/XLSX file **must have exactly 3 fixed column headers** (case-insensitive):

| Column | Type | Format | Example |
|--------|------|--------|---------|
| **start** | Time | `H:MM:SS.MS` or `H:MM:SS` | `0:10:15.50` |
| **end** | Time | `H:MM:SS.MS` or `H:MM:SS` | `0:10:22.75` |
| **category** | String | Any descriptive label | `gesture_wave`, `greeting`, `pause` |

### Optional Additional Columns

You may include any additional columns beyond the required 3. These will be preserved in processing logs but not used by the pipeline.

### Example Structure

```
Row 1 (Header): | start        | end          | category          | notes              | performer |
Row 2:          | 0:00:05.00   | 0:00:10.50   | greeting          | Hello gesture      | Alice     |
Row 3:          | 0:00:15.00   | 0:00:22.75   | wave              | Hand waving        | Alice     |
```

---

## Output Organization

All outputs follow a consistent directory structure:

```
workspaces/
└── video_name/
    ├── chunks_001/                    # (Step 1 output)
    │   ├── video/
    │   ├── audio/
    │   ├── processing_log.csv
    │   ├── segments_log.csv
    │   └── processing_output.txt
    │
    ├── landmarks/                      # (Step 2 output)
    │   ├── segment1_face.csv
    │   ├── segment2_face.csv
    │   └── frames/
    │
    └── plots/                          # (Step 3 output)
        ├── segment1_distances.png
        ├── segment2_distances.png
        └── distance_summary.csv
```

---

## Dependencies Graph

```
KeyPoints.json (filter configuration)
    ↓
video_cutting.py ─────→ CSV/XLSX segments file
    ↓
[chunks_NNN/video/ & chunks_NNN/audio/]
    ↓
extract_landmarks_in_batch.py ─→ Face landmark data
    ↓
[landmarks/ CSVs]
    ↓
analyze_distant_wav_in_batch.py ─→ [plots/ & distance_summary.csv]
    ↓
[Optional] Demo scripts for visualization
```

---

## Module Integration

The noriko-sensei module integrates with:
- **src/utils.py**: Shared utilities (video I/O, visualization)
- **keypoints.json**: Landmark group definitions
- **hand_connections.json**: Hand skeleton structure
- **weights/**: MediaPipe model files

Ensure all dependencies are available before running.

---

## Questions & Support

For issues with:
- **Video processing**: Check FFmpeg installation and PATH
- **Landmark detection**: Review video quality and lighting
- **Plotting**: Verify data exists in landmark CSVs
- **General**: Check `processing_output.txt` for detailed logs

---

**Last Updated**: November 2024
**Tested With**: MediaPipe 0.10.21, OpenCV 4.12.0, Python 3.10+
