# AI Object Detection
A powerful, open-source framework for real-time computer vision and image processing. This project provides fast and accurate object detection capabilities using modern deep-learning models. Designed for developers, researchers, and hobbyists, the toolkit enables seamless integration into applications requiring live video analysis, image recognition, and intelligent automation.

## Key features include:

- Real-time object detection with high accuracy

- Optimized image processing pipelines for performance and reliability

- Modular architecture for easy customization and model swapping

- Support for multiple AI/ML frameworks

- Extensible design suitable for robotics, surveillance, analytics, and more

Perfect for anyone building smart vision systems or experimenting with cutting-edge AI detection models.

# Video Recording: record-pi.py

This script captures video from a Raspberry Pi camera (it can be adjusted to use a USB camera) and saves the recorded videos to the specified folder in .MP4 format.

## --- Parameters:

- output_dir (default: "recorded") – the directory where recorded videos will be saved.

- frame_size (default: (1280, 720)) – the resolution of the recorded video frames.

## --- Output:

Video files will be saved in .MP4 format.

## --- Example usage:

- output_dir = "recorded"
- frame_size = (1280, 720)

# Video Processing: process-video-pi.py

This script processes videos from the recorded folder. For each video, it:

Detects objects specified in a selected list.

Saves images of the detected objects.

Generates a log file in the detected folder.

Once processing is complete, moves the video to the processed folder.

Configuration Parameters:

## --- Target objects and model ---
TARGET_CLASSES = ["person", "bottle", "tv"]  # Objects to detect
MODEL_NAME = "yolov8n.pt"                   # Detection model

## --- Main output folders ---
DETECTED_FRAMES_ROOT = "detected"           # Folder for detected images and logs
PROCESSED_VIDEO_DIR = "processed"           # Folder for videos after processing
CONF_THRESHOLD = 0.5                         # Confidence threshold for detection

## --- Optimization settings ---
SAVE_INTERVAL_FRAMES = 10                    # Save 1 out of every 10 frames with detected objects
PROGRESS_UPDATE_INTERVAL = 10                # Interval for updating progress bar


## --- Folders used:

recorded – input videos to be processed.

detected – stores detected object images and log files.

processed – stores videos after they have been processed.
