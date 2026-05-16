# Face Indexer

⚠️ **This project is in WIP (Work In Progress) state** ⚠️

## Overview

This project is a face detection, recognition, and clustering pipeline designed to process video files and extract, identify, and cluster faces automatically.

## What the Script Does

The main script `faces.py` orchestrates a multi-step pipeline for facial analysis:

1. **Frame Extraction** - Extracts frames from a video file at a specified interval
2. **Face Detection** - Detects and recognizes faces in the extracted frames
3. **Local Clustering** - Groups similar faces together within the video
4. **Global Matching** - Matches local clusters against a global face database
5. **Annotation** - Annotates frames with detected face clusters and labels

## Installation

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python faces.py <video_path>
```

- `<video_path>` (required): Path to the video file to process

### Advanced Options

```bash
python faces.py <video_path> -i <interval>
```

- `-i, --interval` (optional): Frame sampling interval in seconds (default: 10.0)

### Example

```bash
python faces.py video.mp4 -i 5.0
```

This will process `video.mp4`, extracting frames every 5 seconds for face analysis.

## Project Structure

- `faces.py` - Main entry point and pipeline orchestration
- `faces_core.py` - Core functions for extraction, recognition, clustering, and annotation
- `faces_ui.py` - User interface components
- `faces_utils.py` - Utility functions
- `temp/` - Temporary directory for intermediate processing results

## Notes

- The pipeline automatically detects if global labels have already been applied to avoid redundant processing
- To reprocess a video, delete the corresponding local clustering results file
- This is an experimental project with ongoing development

## Status

🚧 **In Development** - Core functionality is under active development and subject to change.
