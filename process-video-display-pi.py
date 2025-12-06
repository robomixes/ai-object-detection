import cv2
from ultralytics import YOLO
import os
import sys
import shutil
from datetime import timedelta

# --- Configuration ---
TARGET_CLASSES = ["person", "bottle", "tv"]
MODEL_NAME = "yolov8n.pt" 
# Main output folders
DETECTED_FRAMES_ROOT = "detected"
PROCESSED_VIDEO_DIR = "processed"
CONF_THRESHOLD = 0.5 

# --- OPTIMIZATION SETTING ---
# Only save 1 out of every 10 consecutive frames that contain a target object.
SAVE_INTERVAL_FRAMES = 10 
# Progress bar update interval
PROGRESS_UPDATE_INTERVAL = 10
# ----------------------------

def process_video_for_detections(video_path):
    # --- Setup ---
    
    # 1. Prepare output directories
    video_filename = os.path.basename(video_path)
    video_name_no_ext = os.path.splitext(video_filename)[0]
    
    # Define paths
    frame_output_dir = os.path.join(DETECTED_FRAMES_ROOT, video_name_no_ext)
    analysis_file_path = os.path.join(frame_output_dir, f"{video_name_no_ext}_analysis.txt")
    
    # Create the frame-saving directory
    if not os.path.exists(frame_output_dir):
        os.makedirs(frame_output_dir)
        print(f"Created frame output folder: {frame_output_dir}")
        
    # Create the processed archive directory
    if not os.path.exists(PROCESSED_VIDEO_DIR):
        os.makedirs(PROCESSED_VIDEO_DIR)

    # 2. Load the YOLO model
    try:
        model = YOLO(MODEL_NAME)
        print(f"YOLO model {MODEL_NAME} loaded.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Please ensure 'ultralytics' is installed.")
        return

    # 3. Load the video and get FPS
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
           # Fallback to a common FPS if property is not available
           fps = 30.0
           print(f"Warning: Could not read video FPS. Defaulting to {fps} FPS.")
        
    frame_time_in_seconds = 1.0 / fps
    print(f"Video FPS detected: {fps}")

    # 4. Map target class names to COCO IDs
    class_name_to_id = {v: k for k, v in model.names.items()}
    target_class_ids = [class_name_to_id.get(name) for name in TARGET_CLASSES if name in class_name_to_id]

    if not target_class_ids:
        print(f"Error: None of the target classes {TARGET_CLASSES} found in model's class list.")
        return
        
    print(f"Target classes detected (COCO IDs): {target_class_ids}")

    # 5. Open the analysis file for writing
    try:
        analysis_file = open(analysis_file_path, 'w')
        analysis_file.write(f"--- YOLOv8 Detection Analysis for: {video_filename} ---\n")
        analysis_file.write(f"Target Classes: {', '.join(TARGET_CLASSES)}\n")
        analysis_file.write(f"Video FPS: {fps:.2f}\n")
        analysis_file.write("-----------------------------------------------------------\n")
        analysis_file.write("FRAME_INDEX | TIME (HH:MM:SS.ms) | DETECTED OBJECTS\n")
        analysis_file.write("-----------------------------------------------------------\n")
    except Exception as e:
        print(f"Error opening analysis file: {e}")
        return


    # --- Processing Loop ---
    frame_count = 0
    detected_frame_count = 0
    
    print("\n--- Starting Video Processing ---")
    
    # Define a window name for the display
    window_name = f"YOLOv8 Live Detection - {video_name_no_ext}"

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # End of video
                break
                
            # Run YOLO inference
            # YOLOv8's 'plot' method automatically draws boxes, class, and confidence.
            results = model.predict(
                source=frame, 
                classes=target_class_ids, 
                conf=CONF_THRESHOLD,
                verbose=False,
                imgsz=640  # Resizes the frame to 640x640 before detection
            )
            
            single_result = results[0]
            boxes = single_result.boxes
            
            # Get the frame with all annotations (boxes, class names, and confidence)
            annotated_frame = single_result.plot() # This is the key line to get the visualization
            
            # --- LIVE DISPLAY ---
            cv2.imshow(window_name, annotated_frame)
            
            # Add a small wait to allow the window to refresh and capture key presses
            # Wait for 1 millisecond. Press 'q' to exit.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # --------------------
            
            # Check if any target objects were detected in the frame
            if len(boxes) > 0:
                # --- OPTIMIZATION CHECK (Saves to Disk/Log) ---
                if frame_count % SAVE_INTERVAL_FRAMES == 0:
                    detected_frame_count += 1
                    
                    # Calculate time in seconds and format
                    total_seconds = frame_count * frame_time_in_seconds
                    time_format = str(timedelta(seconds=total_seconds))
                    
                    # Get list of detected class names
                    # Confidences are stored in boxes.conf, but the class list is enough for the log
                    detected_names = [model.names[int(cls)] for cls in boxes.cls]
                    detected_objects_str = ", ".join(detected_names)

                    # 1. Log to Analysis File
                    analysis_file.write(f"{frame_count:11} | {time_format[:10].zfill(10)} | {detected_objects_str}\n")

                    # 2. Save the annotated frame (the one already generated for display)
                    save_path = os.path.join(frame_output_dir, f"frame_{frame_count:06d}.jpg")
                    cv2.imwrite(save_path, annotated_frame)
                    
                    print(f"Saved frame {frame_count} at {time_format[:10].zfill(10)} with {len(boxes)} detections.")

            frame_count += 1
            
            # Print progress update
            if frame_count % PROGRESS_UPDATE_INTERVAL == 0:
                sys.stdout.write(f"\rFrames processed: {frame_count} | Detected frames saved: {detected_frame_count}")
                sys.stdout.flush()
    
    except Exception as e:
        print(f"\nAn error occurred during video processing: {e}")

    # --- Cleanup and Archiving ---
    cap.release()
    analysis_file.close() # Close the analysis file
    # Ensure the display window is properly closed
    cv2.destroyAllWindows() 
    
    # 5. Move the source video to the 'processed' folder
    try:
        source_path = video_path
        destination_path = os.path.join(PROCESSED_VIDEO_DIR, video_filename)
        shutil.move(source_path, destination_path)
        print(f"\nSuccessfully moved source video to: {destination_path}")
    except Exception as e:
        print(f"Error moving video file: {e}")

    print(f"\n\n--- Processing Complete ---")
    print(f"Total frames processed: {frame_count}")
    print(f"Total detected frames saved: {detected_frame_count}")
    print(f"Analysis log saved to: {analysis_file_path}")


if __name__ == "__main__":
    # --- Main Execution ---
    
    recorded_dir = "recorded"
    
    try:
        video_files = [f for f in os.listdir(recorded_dir) if f.endswith(".mp4")]
    except FileNotFoundError:
        print(f"Error: The directory '{recorded_dir}' was not found.")
        print("Please ensure the recording script was run successfully first.")
        sys.exit(1)

    if not video_files:
        print(f"Error: No .mp4 files found in the '{recorded_dir}' folder to process.")
        print("Please record a video first using the other script.")
        sys.exit(1)
    else:
        # Use the most recently created video file
        video_files.sort(key=lambda f: os.path.getmtime(os.path.join(recorded_dir, f)), reverse=True)
        latest_video = os.path.join(recorded_dir, video_files[0])
        
        print(f"Processing latest video: {latest_video}")
        process_video_for_detections(latest_video)