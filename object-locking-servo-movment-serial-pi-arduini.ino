#include <Servo.h>

Servo panServo; // Pan servo

// Servo configuration
const int SERVO_PIN = 2;
const int SERVO_LEFT = 0;
const int SERVO_RIGHT = 180;
const int SERVO_CENTER = 90;

const int STEP_SIZE = 15;          // Normal movement step
const int SMALL_STEP_SIZE = 1;    // Small correction per command
const unsigned long STEP_DELAY = 3000;

int currentPos = SERVO_CENTER;
int targetPos = SERVO_CENTER;
unsigned long lastStepTime = 0;

// Search mode variables
bool searching = false;
int searchDirection = 1; // 1 = right, -1 = left

void setup() {
  Serial.begin(9600);
  panServo.attach(SERVO_PIN);
  panServo.write(currentPos);
  Serial.println("Gimbal Controller Initialized.");
}

void loop() {
  // Handle incoming serial commands
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    // ===== PAN FULL COMMANDS =====
    if (command.indexOf("PAN LEFT") != -1 && command.indexOf("SMALL") == -1) {
      searching = false;
      targetPos = SERVO_LEFT;
      Serial.println("Command: PAN LEFT (Full)");
    }
    else if (command.indexOf("PAN RIGHT") != -1 && command.indexOf("SMALL") == -1) {
      searching = false;
      targetPos = SERVO_RIGHT;
      Serial.println("Command: PAN RIGHT (Full)");
    }
    else if (command.indexOf("PAN STOP") != -1) {
      searching = false;
      targetPos = currentPos; // Hold position
      Serial.println("Command: PAN STOP (Holding)");
    }

    // ===== PAN SMALL COMMANDS =====
    else if (command.indexOf("PAN_LEFT_SMALL") != -1) {
      searching = false;
      currentPos -= SMALL_STEP_SIZE;
      if (currentPos < SERVO_LEFT) currentPos = SERVO_LEFT;
      panServo.write(currentPos);
      Serial.print("Command: PAN_LEFT_SMALL -> ");
      Serial.println(currentPos);
    }
    else if (command.indexOf("PAN_RIGHT_SMALL") != -1) {
      searching = false;
      currentPos += SMALL_STEP_SIZE;
      if (currentPos > SERVO_RIGHT) currentPos = SERVO_RIGHT;
      panServo.write(currentPos);
      Serial.print("Command: PAN_RIGHT_SMALL -> ");
      Serial.println(currentPos);
    }

    // ===== STATUS COMMANDS =====
    else if (command.equals("LOST")) {
      searching = true;
      searchDirection = 1;
      targetPos = SERVO_CENTER;
      Serial.println("Status: LOST — Starting search sweep.");
    }
    else if (command.equals("FOUND")) {
      searching = false;
      targetPos = currentPos; // Stop movement
      Serial.println("Status: FOUND — Tracking re-acquired.");
    }
    else if (command.equals("TIMEOUT")) {
      searching = false;
      targetPos = SERVO_CENTER;
      Serial.println("Status: TIMEOUT — Servo centered and stopped.");
    }
    else {
      Serial.print("Unknown or Ignored Command: ");
      Serial.println(command);
    }
  }

  // ===== SERVO MOVEMENT LOGIC =====
  unsigned long currentTime = millis();
  if (currentTime - lastStepTime >= STEP_DELAY) {
    lastStepTime = currentTime;

    if (searching) {
      currentPos += searchDirection * STEP_SIZE;

      if (currentPos >= SERVO_RIGHT) {
        currentPos = SERVO_RIGHT;
        searchDirection = -1;
      }
      else if (currentPos <= SERVO_LEFT) {
        currentPos = SERVO_LEFT;
        searchDirection = 1;
      }

      panServo.write(currentPos);

    }
    else if (abs(currentPos - targetPos) > STEP_SIZE) {
      int stepDir = (targetPos > currentPos) ? 1 : -1;
      currentPos += stepDir * STEP_SIZE;

      if ((stepDir > 0 && currentPos > targetPos) || (stepDir < 0 && currentPos < targetPos)) {
        currentPos = targetPos;
      }

      panServo.write(currentPos);
    }
  }
}
