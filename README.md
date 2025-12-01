# AI Object Detection
A powerful, open-source framework for real-time computer vision and image processing. This project provides fast and accurate object detection capabilities using modern deep-learning models. Designed for developers, researchers, and hobbyists, the toolkit enables seamless integration into applications requiring live video analysis, image recognition, and intelligent automation.

## Key features include:

- Real-time object detection with high accuracy

- Optimized image processing pipelines for performance and reliability

- Modular architecture for easy customization and model swapping

- Support for multiple AI/ML frameworks

- Extensible design suitable for robotics, surveillance, analytics, and more

Perfect for anyone building smart vision systems or experimenting with cutting-edge AI detection models.

# 1. Video Recording: record-pi.py

This script captures video from a Raspberry Pi camera (it can be adjusted to use a USB camera) and saves the recorded videos to the specified folder in .MP4 format.

## --- Parameters:

- output_dir (default: "recorded") – the directory where recorded videos will be saved.

- frame_size (default: (1280, 720)) – the resolution of the recorded video frames.

## --- Output:

Video files will be saved in .MP4 format.

## --- Example usage:

- output_dir = "recorded"
- frame_size = (1280, 720)

# 2. Video Processing: process-video-pi.py

This script processes videos from the recorded folder. For each video, it:

Detects objects specified in a selected list.

Saves images of the detected objects.

Generates a log file in the detected folder.

Once processing is complete, moves the video to the processed folder.

Configuration Parameters:

## --- Target objects and model ---
- TARGET_CLASSES = ["person", "bottle", "tv"]  # Objects to detect
- MODEL_NAME = "yolov8n.pt"                   # Detection model

## --- Main output folders ---
- DETECTED_FRAMES_ROOT = "detected"           # Folder for detected images and logs
- PROCESSED_VIDEO_DIR = "processed"           # Folder for videos after processing
- CONF_THRESHOLD = 0.5                         # Confidence threshold for detection

## --- Optimization settings ---
- SAVE_INTERVAL_FRAMES = 10                    # Save 1 out of every 10 frames with detected objects
- PROGRESS_UPDATE_INTERVAL = 10                # Interval for updating progress bar


## --- Folders used:

recorded – input videos to be processed.

detected – stores detected object images and log files.

processed – stores videos after they have been processed.

## --------------- Log file example --------------
- YOLOv8 Detection Analysis for: video_20251130_224142.mp4 
- Target Classes: person, bottle, tv 
- Video FPS: 30.00

-----------------------------------------------------------
FRAME_INDEX | TIME (HH:MM:SS.ms) | DETECTED OBJECTS
-----------------------------------------------------------
         30 | 0000:00:01 | person
         50 | 0:00:01.66 | person
         60 | 0000:00:02 | person
         70 | 0:00:02.33 | person
         80 | 0:00:02.66 | person
         90 | 0000:00:03 | person
        100 | 0:00:03.33 | person
        350 | 0:00:11.66 | tv, person
        360 | 0000:00:12 | tv

# 3. Object Detection and Locking Mechanism

This section contains three main scripts:

- object-locking-pi

- object-locking-mqtt-receiver-pi

- object-locking-servo-movement-serial-pi-arduino

The idea behind this project is to define a list of target objects (e.g., person, car, TV, etc.).
The camera will detect the defined objects, and using your mouse you can lock one object as the main focus. Once locked, the camera will attempt to keep that object in view.

Running object-locking-pi allows you to lock a detected object and publish MQTT messages. These messages can be received by other scripts to take further actions.

The second script, object-locking-mqtt-receiver-pi, is an example of receiving MQTT messages from the object-locking-pi script and then sending serial commands based on the object’s status.

The last script, object-locking-servo-movement-serial-pi-arduino, shows how to receive the serial commands, process them, and move a servo to help keep the locked object centered.



