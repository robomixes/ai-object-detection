from ultralytics import YOLO
import cv2
from picamera2 import Picamera2
import numpy as np
import math
import paho.mqtt.client as mqtt
import json
import time

# --- Global: Define Tracking State and Target ---
TARGET_CLASSES = ["person", "bottle", "tv"]
MIN_CONFIDENCE = 0.4

# --- Configuration Parameter ---
AUTO_FOCUS_ON_BIGGEST = False 

# --- ROI Configuration ---
# ROI state variables
ROI_MODE = False        # True if the user is currently drawing the ROI
ROI_ACTIVE = False      # True if a valid ROI has been selected and should be used for filtering
ROI_START_POINT = None  # (x, y) of the first click
ROI_END_POINT = None    # (x, y) of the current or final release point

# --- MQTT Setup ---
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_STATUS_TOPIC = "tracker/move"    
MQTT_TRACKING_TOPIC = "tracker/status" 
MQTT_CLIENT_ID = "YOLO_Tracker"
mqtt_client = None 

# State variables for mouse control and tracking
FOCUS_MODE = False
OBJECT_RECENTLY_LOST = False  
FOCUSED_OBJECT_CLS = -1
FOCUSED_OBJECT_BOX_COORDS = None  
FOCUSED_OBJECT_CONF = 0.0
FOCUSED_OBJECT_TEMPLATE = None    

# --- Tracking Parameters ---
MIN_SIMILARITY_MATCH = 0.40          
TEMPLATE_UPDATE_THRESHOLD = 0.90    
MAX_PIXEL_SHIFT = 150              
MAX_LOST_FRAMES = 150              
lost_frames_counter = 0             
AUTO_FOCUS_ACTIVE = False           

# --- Load YOLOv8 Model ---
try:
    model = YOLO("yolov8n.pt")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# --- Initialize PiCamera2 ---
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam2.configure(preview_config)
picam2.start()

# --- Calculate Frame Center ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CENTER_X = FRAME_WIDTH // 2
CENTER_Y = FRAME_HEIGHT // 2

# ----------------------------------------------------
# ## 🔌 MQTT Functions
# ----------------------------------------------------

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ MQTT Connected successfully.")
    else:
        print(f"❌ MQTT Connection failed with code {rc}.")

def connect_mqtt():
    global mqtt_client
    try:
        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, MQTT_CLIENT_ID)
        mqtt_client.on_connect = on_connect
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start() 
    except Exception as e:
        print(f"Could not connect to MQTT broker: {e}")
        mqtt_client = None

def publish_move_status(offset_x, offset_y, pan_cmd, tilt_cmd):
    """
    Publishes the object position and gimbal instructions via MQTT.
    """
    global mqtt_client
    if mqtt_client is None: return

    payload = {
        "timestamp": time.time(),
        "offset_x": offset_x,
        "offset_y": offset_y,
        "pan_command": pan_cmd,
        "tilt_command": tilt_cmd
    }
    
    try:
        # print(f"-> MQTT MOVE: X:{offset_x}, Y:{offset_y}, Pan:{pan_cmd}, Tilt:{tilt_cmd}")
        mqtt_client.publish(MQTT_STATUS_TOPIC, json.dumps(payload), qos=0)
    except Exception as e:
        print(f"Error publishing MOVE message: {e}")

def publish_tracking_status(status):
    global mqtt_client
    if mqtt_client is None: return
    
    payload = {
        "timestamp": time.time(),
        "status": status
    }
    
    try:
        print(f"-> MQTT TRACKING STATUS: {status}")
        mqtt_client.publish(MQTT_TRACKING_TOPIC, json.dumps(payload), qos=0)
    except Exception as e:
        print(f"Error publishing TRACKING status: {e}")

# ----------------------------------------------------
# ## 🎯 Focus Helper Function
# ----------------------------------------------------

def start_focus(frame, x1, y1, x2, y2, class_id, conf):
    # Global declaration for variables that will be modified
    global FOCUS_MODE, FOCUSED_OBJECT_CLS, FOCUSED_OBJECT_BOX_COORDS, FOCUSED_OBJECT_CONF, FOCUSED_OBJECT_TEMPLATE, OBJECT_RECENTLY_LOST, lost_frames_counter

    OBJECT_RECENTLY_LOST = False
    lost_frames_counter = 0

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(FRAME_WIDTH, x2), min(FRAME_HEIGHT, y2)
    
    template_rgb = frame[y1:y2, x1:x2].copy()
    
    if template_rgb.size == 0:
        print("Error: Cropped area for focus is empty.")
        return False
        
    FOCUSED_OBJECT_TEMPLATE = cv2.cvtColor(template_rgb, cv2.COLOR_BGR2GRAY)
    
    FOCUSED_OBJECT_BOX_COORDS = (x1, y1, x2, y2)
    FOCUSED_OBJECT_CLS = class_id
    FOCUSED_OBJECT_CONF = conf
    FOCUS_MODE = True
    print(f"\n[FOCUS ACQUIRED] Tracking {model.names[class_id]}. Initial Template Created.")
    return True

# ----------------------------------------------------
# ## 🖱️ Mouse Callback Function (ROI & Focus Control)
# ----------------------------------------------------

def mouse_callback(event, x, y, flags, param):
    # --- FIXED: Global declarations moved to the top of the function ---
    global FOCUS_MODE, AUTO_FOCUS_ACTIVE, ROI_MODE, ROI_ACTIVE, ROI_START_POINT, ROI_END_POINT
    global OBJECT_RECENTLY_LOST, FOCUSED_OBJECT_CLS, FOCUSED_OBJECT_BOX_COORDS, FOCUSED_OBJECT_TEMPLATE, lost_frames_counter
    # --------------------------------------------------------------------
    
    current_boxes = param[0]
    frame = param[1] 

    # --- ROI Handling (Middle-Click) ---
    if event == cv2.EVENT_MBUTTONDOWN:  # CHANGED from RBUTTONDOWN
        # If ROI is active, middle-click clears it
        if ROI_ACTIVE or ROI_MODE:
            ROI_ACTIVE = False
            ROI_MODE = False
            ROI_START_POINT = None
            ROI_END_POINT = None
            print("[ROI CLEARED] Detection scope returned to full frame.")
        # If ROI is not active, middle-click starts drawing
        else:
            ROI_MODE = True
            ROI_START_POINT = (x, y)
            print("[ROI MODE] Middle-click to start drawing. Release to finalize.")
        
        # Clear any active FOCUS mode when changing ROI
        if FOCUS_MODE:
            FOCUS_MODE = False
            OBJECT_RECENTLY_LOST = False
            FOCUSED_OBJECT_CLS = -1
            FOCUSED_OBJECT_BOX_COORDS = None
            FOCUSED_OBJECT_TEMPLATE = None
            lost_frames_counter = 0
            publish_tracking_status("MANUAL_STOP")

    elif event == cv2.EVENT_MOUSEMOVE:
        if ROI_MODE:
            ROI_END_POINT = (x, y) # Update the end point for live drawing

    elif event == cv2.EVENT_MBUTTONUP:  # CHANGED from RBUTTONUP
        if ROI_MODE and ROI_START_POINT is not None:
            ROI_MODE = False
            ROI_END_POINT = (x, y)
            
            # Ensure x1 < x2 and y1 < y2 for proper ROI definition
            x1 = min(ROI_START_POINT[0], ROI_END_POINT[0])
            y1 = min(ROI_START_POINT[1], ROI_END_POINT[1])
            x2 = max(ROI_START_POINT[0], ROI_END_POINT[0])
            y2 = max(ROI_START_POINT[1], ROI_END_POINT[1])
            
            # Check if the ROI is a reasonable size (e.g., > 10x10)
            if (x2 - x1) > 10 and (y2 - y1) > 10:
                ROI_ACTIVE = True
                ROI_START_POINT = (x1, y1)
                ROI_END_POINT = (x2, y2)
                print(f"[ROI ACTIVE] Set to: ({x1}, {y1}) to ({x2}, {y2}).")
            else:
                ROI_ACTIVE = False
                ROI_START_POINT = None
                ROI_END_POINT = None
                print("[ROI CANCELED] Area too small.")

    # --- FOCUS Handling (Left-Click) ---
    if event == cv2.EVENT_LBUTTONDOWN:
        
        if AUTO_FOCUS_ACTIVE:
            AUTO_FOCUS_ACTIVE = False 

        # 1. Clear Focus/Seeking state
        FOCUS_MODE = False
        OBJECT_RECENTLY_LOST = False
        FOCUSED_OBJECT_CLS = -1
        FOCUSED_OBJECT_BOX_COORDS = None
        FOCUSED_OBJECT_TEMPLATE = None
        lost_frames_counter = 0
        print("\n[FOCUS CLEARED] Returning to General Tracking.")
        
        publish_tracking_status("MANUAL_STOP")
        
        # 2. If the click lands on an object, start focus on that object
        min_dist_to_click = float('inf')
        best_match_at_click = None

        for box_data in current_boxes:
            x1, y1, x2, y2, class_id, conf, class_name = box_data
            
            # Filter: Check if the clicked object is inside the active ROI
            if ROI_ACTIVE:
                roi_x1, roi_y1 = ROI_START_POINT
                roi_x2, roi_y2 = ROI_END_POINT
                
                # Check if the click is within the ROI
                if not (roi_x1 <= x <= roi_x2 and roi_y1 <= y <= roi_y2):
                    continue
            
            if x1 <= x <= x2 and y1 <= y <= y2 and class_name in TARGET_CLASSES and conf > MIN_CONFIDENCE:
                obj_center_x = (x1 + x2) // 2
                obj_center_y = (y1 + y2) // 2
                distance = math.sqrt((obj_center_x - x)**2 + (obj_center_y - y)**2)
                
                if distance < min_dist_to_click:
                    min_dist_to_click = distance
                    best_match_at_click = box_data

        if best_match_at_click:
            x1, y1, x2, y2, class_id, conf, _ = best_match_at_click
            start_focus(frame, x1, y1, x2, y2, class_id, conf)


# --- Setup Mouse Handler ---
WINDOW_NAME = "YOLOv8 Object Detection (L-Click FOCUS, M-Click ROI)"
cv2.namedWindow(WINDOW_NAME)
print(f"Starting detection. Auto-focus is {'ON' if AUTO_FOCUS_ON_BIGGEST else 'OFF'}.")
# Updated instruction text:
print("Controls: L-Click on an object to focus/untrack. M-Click and drag to define ROI. M-Click again to clear ROI. Press 'q' to quit.")

# --- Connect MQTT ---
connect_mqtt()

# ----------------------------------------------------
# ## 📹 Main Detection Loop
# ----------------------------------------------------

while True:
    frame = picam2.capture_array()
    
    # 1. CROP FRAME TO ROI IF ACTIVE
    x1_roi, y1_roi, x2_roi, y2_roi = 0, 0, FRAME_WIDTH, FRAME_HEIGHT
    
    if ROI_ACTIVE and not ROI_MODE:
        x1_roi, y1_roi = ROI_START_POINT
        x2_roi, y2_roi = ROI_END_POINT
        
        # Clamp coordinates to ensure they are within frame bounds
        x1_roi, y1_roi = max(0, x1_roi), max(0, y1_roi)
        x2_roi, y2_roi = min(FRAME_WIDTH, x2_roi), min(FRAME_HEIGHT, x2_roi)
        
        # Use the cropped frame for YOLO detection
        frame_roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]
        
        if frame_roi.size > 0:
             results = model(frame_roi)
        else:
             results = model(frame) # Fallback to full frame
    else:
        results = model(frame)
        
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    current_boxes_data = [] 

    # --- AUTO-SELECTION LOGIC (Filter by ROI) ---
    if AUTO_FOCUS_ON_BIGGEST and not AUTO_FOCUS_ACTIVE and not FOCUS_MODE:
        best_area = 0
        best_object_data = None
        
        for result in results:
            for box in result.boxes:
                # Coordinates are relative to the input frame (either full or ROI)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[class_id]

                # Convert coords back to full frame if ROI was used
                if ROI_ACTIVE and not ROI_MODE:
                    x1 += x1_roi
                    y1 += y1_roi
                    x2 += x1_roi
                    y2 += y1_roi
                    
                if class_name in TARGET_CLASSES and conf > MIN_CONFIDENCE:
                    area = (x2 - x1) * (y2 - y1)
                    if area > best_area:
                        best_area = area
                        best_object_data = (x1, y1, x2, y2, class_id, conf)

        if best_object_data is not None:
            x1, y1, x2, y2, class_id, conf = best_object_data
            if start_focus(frame, x1, y1, x2, y2, class_id, conf):
                AUTO_FOCUS_ACTIVE = True 
                
    # --- Check for Seeking Mode Timeout ---
    if OBJECT_RECENTLY_LOST:
        lost_frames_counter += 1
        if lost_frames_counter > MAX_LOST_FRAMES:
            FOCUS_MODE = False
            OBJECT_RECENTLY_LOST = False
            FOCUSED_OBJECT_CLS = -1
            FOCUSED_OBJECT_TEMPLATE = None
            lost_frames_counter = 0
            print("[TIMEOUT] Seeking timeout. Returning to general track.")
            publish_tracking_status("TIMEOUT")
    
    # --- Tracking and Drawing Logic ---
    if FOCUS_MODE and FOCUSED_OBJECT_TEMPLATE is not None:
        
        highest_similarity_score = -1.0
        best_match_box = None
        found_focused_object = False
        
        search_candidates = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Coordinates are relative to the input frame (full or ROI)
                x1_rel, y1_rel, x2_rel, y2_rel = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = x1_rel, y1_rel, x2_rel, y2_rel # Default to relative
                
                # Convert coords back to full frame if ROI was used
                if ROI_ACTIVE and not ROI_MODE:
                    x1 += x1_roi
                    y1 += y1_roi
                    x2 += x1_roi
                    y2 += y1_roi
                    
                if class_id == FOCUSED_OBJECT_CLS and conf > MIN_CONFIDENCE:
                    search_candidates.append((x1, y1, x2, y2)) # Store absolute coords

        # --- Perform Template Matching (On Absolute Coordinates) ---
        for new_x1, new_y1, new_x2, new_y2 in search_candidates:
            
            # Proximity Check (uses absolute coordinates)
            if not OBJECT_RECENTLY_LOST and FOCUSED_OBJECT_BOX_COORDS is not None:
                last_x1, last_y1, last_x2, last_y2 = FOCUSED_OBJECT_BOX_COORDS
                last_center_x = (last_x1 + last_x2) // 2
                last_center_y = (last_y1 + last_y2) // 2
                new_center_x = (new_x1 + new_x2) // 2
                new_center_y = (new_y1 + new_y2) // 2
                distance = math.sqrt((new_center_x - last_center_x)**2 + (new_center_y - last_center_y)**2)
                if distance > MAX_PIXEL_SHIFT:
                    continue 

            candidate_image_gray = frame_gray[new_y1:new_y2, new_x1:new_x2]
            if candidate_image_gray.shape[0] < 5 or candidate_image_gray.shape[1] < 5:
                continue 
            
            try:
                resized_candidate_gray = cv2.resize(
                    candidate_image_gray, 
                    (FOCUSED_OBJECT_TEMPLATE.shape[1], FOCUSED_OBJECT_TEMPLATE.shape[0])
                )
                result_matrix = cv2.matchTemplate(resized_candidate_gray, FOCUSED_OBJECT_TEMPLATE, cv2.TM_CCOEFF_NORMED)
                similarity_score = result_matrix[0, 0]

                if similarity_score > highest_similarity_score:
                    highest_similarity_score = similarity_score
                    best_match_box = (new_x1, new_y1, new_x2, new_y2)
                    
            except cv2.error:
                continue

        # --- Update State Based on Match Score ---
        if highest_similarity_score > MIN_SIMILARITY_MATCH:
            found_focused_object = True
        
        if found_focused_object:
            # SUCCESS: Object found/re-acquired
            FOCUSED_OBJECT_BOX_COORDS = best_match_box
            
            if OBJECT_RECENTLY_LOST:
                OBJECT_RECENTLY_LOST = False
                lost_frames_counter = 0
                publish_tracking_status("FOUND") 

            # Adaptive Template Update
            x1, y1, x2, y2 = FOCUSED_OBJECT_BOX_COORDS
            if highest_similarity_score > TEMPLATE_UPDATE_THRESHOLD:
                template_rgb = frame[y1:y2, x1:x2].copy()
                FOCUSED_OBJECT_TEMPLATE = cv2.cvtColor(template_rgb, cv2.COLOR_BGR2GRAY)
            
            # --- CALCULATE AND PUBLISH INSTRUCTIONS ---
            object_center_x = (x1 + x2) // 2
            object_center_y = (y1 + y2) // 2
            
            offset_x = int(object_center_x - CENTER_X)
            offset_y = int(object_center_y - CENTER_Y)
            
            gimbal_pan_cmd = "PAN LEFT" if offset_x > 10 else ("PAN RIGHT" if offset_x < -10 else "PAN STOP")
            gimbal_tilt_cmd = "TILT UP" if offset_y > 10 else ("TILT DOWN" if offset_y < -10 else "TILT STOP")
            
            publish_move_status(offset_x, offset_y, gimbal_pan_cmd, gimbal_tilt_cmd)
            
            # --- Drawing ---
            color = (0, 255, 0) 
            status_text = f"FOCUS Score:{highest_similarity_score:.2f} X:{offset_x}, Y:{offset_y}"
            gimbal_instructions = f"GIMBAL: {gimbal_pan_cmd}, {gimbal_tilt_cmd} (MQTT)"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4) 
            cv2.putText(frame, f"FOCUS {model.names[FOCUSED_OBJECT_CLS]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, status_text, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, gimbal_instructions, (x1, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.circle(frame, (object_center_x, object_center_y), 5, (0, 255, 255), -1) 
            
        # Handle temporary loss (start seeking)
        else:
            if not OBJECT_RECENTLY_LOST:
                OBJECT_RECENTLY_LOST = True
                publish_tracking_status("LOST") 
                publish_move_status(0, 0, "PAN STOP", "TILT STOP") 
             
            status_text = f"SEEKING... ({lost_frames_counter} frames)"
            color = (0, 165, 255) 
            
            if FOCUSED_OBJECT_BOX_COORDS is not None:
                x1, y1, x2, y2 = FOCUSED_OBJECT_BOX_COORDS
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


    # --- General Tracking Mode (Filter and Draw) ---
    if not FOCUS_MODE:
        for result in results:
            for box in result.boxes:
                # Coordinates are relative to the input frame (either full or ROI)
                x1_rel, y1_rel, x2_rel, y2_rel = box.xyxy[0].cpu().numpy().astype(int)
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[class_id]

                # Convert coords back to full frame if ROI was used
                x1, y1, x2, y2 = x1_rel, y1_rel, x2_rel, y2_rel
                if ROI_ACTIVE and not ROI_MODE:
                    x1 += x1_roi
                    y1 += y1_roi
                    x2 += x1_roi
                    y2 += y1_roi

                if class_name in TARGET_CLASSES and conf > MIN_CONFIDENCE:
                    current_boxes_data.append((x1, y1, x2, y2, class_id, conf, class_name))
                    
                    object_center_x = (x1 + x2) // 2
                    object_center_y = (y1 + y2) // 2
                    offset_x = object_center_x - CENTER_X
                    offset_y = object_center_y - CENTER_Y
                    position_text = f"X:{offset_x}, Y:{offset_y}"
                    
                    # Draw detection box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) 
                    # Display class and confidence
                    cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(frame, position_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # --- Draw ROI Rectangle (FIXED SCOPE) ---
    if ROI_ACTIVE or ROI_MODE:
        
        p1 = None # Initialize coordinates to ensure they are defined
        p2 = None
        
        # Determine current points (for drawing a live box or the finalized box)
        if ROI_MODE and ROI_START_POINT is not None and ROI_END_POINT is not None:
            # Drawing the box live
            p1, p2 = ROI_START_POINT, ROI_END_POINT
            roi_color = (0, 255, 255) # Yellow while drawing
            roi_thickness = 1
            roi_text = "Drawing ROI"
        
        elif ROI_ACTIVE and ROI_START_POINT is not None and ROI_END_POINT is not None:
            # Drawing the finalized box
            p1, p2 = ROI_START_POINT, ROI_END_POINT
            roi_color = (255, 0, 255) # Magenta for active ROI
            roi_thickness = 2
            roi_text = "ROI ACTIVE (M-Click to Clear)"
        
        # Execute the drawing ONLY if p1 and p2 were set successfully inside the inner blocks
        if p1 is not None and p2 is not None:
            cv2.rectangle(frame, p1, p2, roi_color, roi_thickness)
            cv2.putText(frame, roi_text, (min(p1[0], p2[0]), max(p1[1], p2[1]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1)


    # --- Draw Center of Frame (Red Crosshair) ---
    crosshair_color = (0, 0, 255) 
    crosshair_size = 30           
    thickness = 2               
    
    cv2.line(frame, (CENTER_X - crosshair_size, CENTER_Y), (CENTER_X + crosshair_size, CENTER_Y), crosshair_color, thickness)
    cv2.line(frame, (CENTER_X, CENTER_Y - crosshair_size), (CENTER_X, CENTER_Y + crosshair_size), crosshair_color, thickness)

    # --- Set Mouse Callback ---
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, (current_boxes_data, frame))
    
    # Show frame
    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Cleanup ---
if mqtt_client:
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    print("MQTT Disconnected.")
    
picam2.stop()
cv2.destroyAllWindows()