import paho.mqtt.client as mqtt
import time
import json
import serial
import threading
import sys # Import sys for cleaner exit on failure

# --- MQTT Configuration ---
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_MOVE_TOPIC = "tracker/move"
MQTT_STATUS_TOPIC = "tracker/status"
MQTT_CLIENT_ID = "Gimbal_Controller"

# --- Serial Configuration ---
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600
SERIAL_TIMEOUT = 1

# --- Global Variables ---
arduino = None
tracking_lost = False
last_command = None
repeat_count = 0
REQUIRED_REPEATS = 3
STABLE_ZONE = 10        # ±20 = stable zone
busy = False            # flag: waiting for stop

# -----------------------------------------------------------------
# --- Serial Write Function ---
# -----------------------------------------------------------------
def send_to_arduino(data_string):
    global arduino
    if arduino is None or not arduino.is_open:
        # This print will occur if serial is not initialized, but MQTT will still receive messages.
        print("❌ Serial not ready. Command not sent.")
        return
    try:
        arduino.write((data_string + '\n').encode('utf-8'))
        print(f"-> SERIAL SENT: {data_string}")
    except Exception as e:
        print(f"❌ Serial write error: {e}")

def handle_command_with_stop(command):
    """
    Send a command, wait 1s, send PAN STOP.
    Blocks new commands until finished.
    """
    global busy
    # Only proceed if serial is potentially connected, otherwise the lock serves no purpose.
    # We still manage 'busy' to regulate message processing, even without serial.
    busy = True
    send_to_arduino(command)
    time.sleep(1)
    send_to_arduino("PAN STOP")
    busy = False

# -----------------------------------------------------------------
# --- MQTT Callbacks ---
# -----------------------------------------------------------------
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ MQTT Connected.")
        client.subscribe(MQTT_MOVE_TOPIC)
        client.subscribe(MQTT_STATUS_TOPIC)
    else:
        print(f"❌ MQTT connection failed with code {rc}")

def on_message(client, userdata, msg):
    global tracking_lost, last_command, repeat_count, busy

    try:
        payload = json.loads(msg.payload.decode())
        topic = msg.topic

        # --- Status Messages ---
        if topic == MQTT_STATUS_TOPIC:
            status = payload.get("status", "UNKNOWN")
            if status == "FOUND":
                tracking_lost = False
                send_to_arduino("FOUND")
                time.sleep(0.2)
                send_to_arduino("PAN STOP")
            elif status == "LOST":
                tracking_lost = True
                send_to_arduino("LOST")
            elif status == "TIMEOUT":
                send_to_arduino("TIMEOUT")
            return

        # --- Move Messages ---
        if topic == MQTT_MOVE_TOPIC:
            if tracking_lost:
                return
            if busy:
                # currently executing a previous command → ignore new commands
                print("⏱️ Busy, ignoring new command until stop is sent.")
                return

            offset_x = payload.get("offset_x", 0)
            pan_cmd = payload.get("pan_command", "PAN STOP")
            tilt_cmd = payload.get("tilt_command", "TILT STOP")

            # --- Stable zone ---
            if -STABLE_ZONE <= offset_x <= STABLE_ZONE:
                print(f"🟢 Stable zone (-{STABLE_ZONE} to +{STABLE_ZONE}) → no movement")
                last_command = None
                repeat_count = 0
                return

            # --- Correct small-step direction ---
            if offset_x > STABLE_ZONE:
                # Object is to the left → move servo left
                gimbal_command = "PAN_LEFT_SMALL"
            elif offset_x < -STABLE_ZONE:
                # Object is to the right → move servo right
                gimbal_command = "PAN_RIGHT_SMALL"
            else:
                gimbal_command = f"{pan_cmd} {tilt_cmd}"

            # --- Repeat filter to avoid jitter ---
            if gimbal_command == last_command:
                repeat_count += 1
            else:
                repeat_count = 1
                last_command = gimbal_command

            if repeat_count >= REQUIRED_REPEATS:
                # Execute command with stop in a separate thread
                # The send_to_arduino function handles the case where the serial port is not open.
                threading.Thread(target=handle_command_with_stop, args=(gimbal_command,)).start()
                repeat_count = 0
            else:
                print(f"Waiting for stability {repeat_count}/{REQUIRED_REPEATS} → {gimbal_command}")

    except json.JSONDecodeError:
        print(f"⚠️ Invalid JSON: {msg.payload.decode()}")
    except Exception as e:
        print(f"❌ Error processing message: {e}")

# -----------------------------------------------------------------
# --- Main ---
# -----------------------------------------------------------------
def main():
    global arduino
    print("Starting Gimbal MQTT Receiver...")

    ## 1. Initialize MQTT First
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=MQTT_CLIENT_ID)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        # Attempt to connect to the broker
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"❌ MQTT connection failed: {e}")
        # If MQTT fails, there's no point in running, so exit.
        sys.exit(1)

    ## 2. Initialize Serial Connection (Optional for initial run)
    try:
        print(f"Attempting to connect to serial port {SERIAL_PORT}...")
        # If this fails, 'arduino' remains None, but the program continues.
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        time.sleep(2) # Wait for Arduino reset
        print("✅ Serial connected.")
    except Exception as e:
        print(f"⚠️ Serial failed: {e}. Program will proceed, but commands will not be sent to device.")
        # DO NOT exit here. This is the main change.

    ## 3. Start MQTT Loop
    try:
        print("Starting MQTT loop forever...")
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n🛑 Exiting.")
    finally:
        client.loop_stop()
        client.disconnect()
        # Clean up serial connection only if it was successfully opened
        if arduino and arduino.is_open:
            arduino.close()
            print("Serial closed.")

if __name__ == "__main__":
    main()