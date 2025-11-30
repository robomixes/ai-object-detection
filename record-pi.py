import cv2
from picamera2 import Picamera2
import time
from datetime import datetime
import os
import sys

def record_video_on_keypress():
    # --- Configuration ---
    output_dir = "recorded"
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            return
    
    # Generate a unique filename using a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_dir, f"video_{timestamp}.mp4")

    # --- Picamera2 Initialization ---
    try:
        picam2 = Picamera2()
        
        # Configure the camera for video: 1280x720 is a good standard HD resolution
        frame_size = (1280, 720)
        config = picam2.create_video_configuration(main={"size": frame_size, "format": "RGB888"})
        picam2.configure(config)
        
        # Start the camera capture
        picam2.start()
        print("Camera started.")
        time.sleep(1) # Allow camera sensor to initialize
        
    except Exception as e:
        print(f"Error initializing Picamera2: {e}")
        # Check for the common "in use" error and provide a helpful tip
        if "Pipeline handler in use by another process" in str(e):
             print("TIP: The camera is likely in use by another program. Please kill that process or reboot the Pi.")
        return

    # --- OpenCV Video Writer Setup ---
    # Define the video codec (H.264 is common and good)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    fps = 30.0 # Framerate
    
    # Initialize the VideoWriter object
    out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)
    
    if not out.isOpened():
        print(f"Error: VideoWriter could not be opened for file {output_filename}.")
        picam2.stop()
        return

    # --- Recording Loop ---
    print("\n--- Recording Started ---")
    print(f"Saving video to: {output_filename}")
    print("Press the **q** key or the **Enter** key while the video window is focused to **STOP** recording.")
    print("Or press **Ctrl+C** in the terminal to stop.")
    
    try:
        while True:
            # Capture the frame from the camera as a NumPy array
            frame = picam2.capture_array()
            
            # **CRITICAL FIX:** Manual RGB to BGR conversion.
            # This avoids the error "module 'cv2' has no attribute 'COLOR_RGB2_BGR'"
            # Picamera2 outputs RGB, but OpenCV's VideoWriter expects BGR.
            # Slicing [:, :, ::-1] reverses the color channels.
            frame_bgr = frame[:, :, ::-1]
            
            # Write the frame to the video file
            out.write(frame_bgr)
            
            # Display the frame in a window for user feedback and keyboard input
            cv2.imshow('Recording - Press q or Enter to STOP', frame_bgr)
            
            # Check for a keyboard press to stop (waitKey(1) waits 1ms)
            key = cv2.waitKey(1) & 0xFF 
            
            # Check for 'q' (standard quit) or 13 (ASCII for Enter key)
            if key == ord('q') or key == 13: 
                 print("\nStop key pressed. Stopping recording...")
                 break 
            
    except KeyboardInterrupt:
        print("\n**Ctrl+C** detected. Stopping recording...")
    except Exception as e:
        print(f"An unexpected error occurred during recording: {e}")

    # --- Cleanup ---
    print("Releasing resources...")
    # Release the VideoWriter
    out.release()
    # Stop the camera and close the OpenCV window
    picam2.stop()
    cv2.destroyAllWindows()
    
    print(f"Recording finished. Video saved to: {output_filename}")


if __name__ == "__main__":
    record_video_on_keypress()




