import cv2
import mediapipe as mp
import pyautogui
import time
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Screen size
screen_width, screen_height = pyautogui.size()
print("Screen:", screen_width, screen_height)


def pinch_distance(hand_landmarks, finger_tip):
    """Distance between thumb tip and another fingertip."""
    thumb = hand_landmarks[4]
    finger = hand_landmarks[finger_tip]
    return math.sqrt(
        (thumb.x - finger.x) ** 2 +
        (thumb.y - finger.y) ** 2
    )


def is_click(hand_landmarks):
    # Thumb + index finger = left-click
    return pinch_distance(hand_landmarks, 8) < 0.05


def is_right_click(hand_landmarks):
    # Thumb + middle finger = right-click
    return pinch_distance(hand_landmarks, 12) < 0.05


def is_finger_up(hand_landmarks, tip, pip):
    return hand_landmarks[tip].y < hand_landmarks[pip].y


def is_thumb_up(hand_landmarks):
    return hand_landmarks[4].x > hand_landmarks[3].x


def detect_gesture(thumb_up, index_up, middle_up, ring_up, pinky_up):
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "POINTING"
    elif not index_up and not middle_up and not ring_up and not pinky_up:
        return "FIST"
    elif index_up and middle_up and ring_up and pinky_up:
        return "OPEN HAND"
    elif index_up and middle_up and not ring_up and not pinky_up:
        return "TWO FINGERS"
    return "UNKNOWN"


# MediaPipe hand landmark model
MODEL_PATH = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)


# Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Error: Could not access webcam.")
    raise SystemExit(1)

# Mouse movement state
prev_x = 0
prev_y = 0
smoothness = 0.3

# Click state
left_pinching = False
right_pinching = False
last_index_pinch_time = None
pending_single_click = False
DOUBLE_PINCH_WINDOW = 0.4  # seconds allowed between two pinches

# Scroll state
prev_scroll_y = None

# Pause/resume: hold an open hand for 1 second
paused = False
pause_gesture_start = None
last_pause_toggle = 0
pause_hold_seconds = 1.0
pause_cooldown = 1.5

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to access webcam.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )
        results = detector.detect(mp_image)

        status_text = "PAUSED" if paused else "ACTIVE"

        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]

            # Draw landmarks
            h, w, _ = frame.shape
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            index_tip = hand_landmarks[8]
            index_up = is_finger_up(hand_landmarks, 8, 6)
            middle_up = is_finger_up(hand_landmarks, 12, 10)
            ring_up = is_finger_up(hand_landmarks, 16, 14)
            pinky_up = is_finger_up(hand_landmarks, 20, 18)
            thumb_up = is_thumb_up(hand_landmarks)

            # Open palm (four fingers extended) toggles pause/resume when held.
            open_palm = index_up and middle_up and ring_up and pinky_up
            now = time.time()

            if open_palm:
                if pause_gesture_start is None:
                    pause_gesture_start = now

                if (
                    now - pause_gesture_start >= pause_hold_seconds
                    and now - last_pause_toggle >= pause_cooldown
                ):
                    paused = not paused
                    last_pause_toggle = now
                    pause_gesture_start = None

                    # Reset click/scroll states so controls don't stick.
                    left_pinching = False
                    right_pinching = False
                    last_index_pinch_time = None
                    pending_single_click = False
                    prev_scroll_y = None
                    if paused:
                        pyautogui.mouseUp()
            else:
                pause_gesture_start = None

            if not paused:
                # Thumb + index pinch: one pinch = left-click; two quick pinches = double-click.
                click = is_click(hand_landmarks)
                now = time.time()

                # If a first pinch is still waiting and the double-pinch window expires,
                # complete it as a normal single-click.
                if pending_single_click and last_index_pinch_time is not None:
                    if now - last_index_pinch_time > DOUBLE_PINCH_WINDOW:
                        pyautogui.click()
                        pending_single_click = False
                        last_index_pinch_time = None

                if click and not left_pinching:
                    if (
                        pending_single_click
                        and last_index_pinch_time is not None
                        and now - last_index_pinch_time <= DOUBLE_PINCH_WINDOW
                    ):
                        pyautogui.doubleClick(interval=0.12)
                        pending_single_click = False
                        last_index_pinch_time = None
                    else:
                        pending_single_click = True
                        last_index_pinch_time = now
                    left_pinching = True
                elif not click:
                    left_pinching = False

                # Right-click: thumb + middle pinch.
                right_click = is_right_click(hand_landmarks)
                if right_click and not right_pinching:
                    pyautogui.rightClick()
                    right_pinching = True
                elif not right_click:
                    right_pinching = False

                # Move mouse only when pointing (not during other pinch gestures).
                pointing = (
                    index_up and not middle_up and
                    not ring_up and not pinky_up
                )
                if pointing:
                    mouse_x = int(index_tip.x * screen_width)
                    mouse_y = int(index_tip.y * screen_height)
                    smooth_x = int(prev_x + (mouse_x - prev_x) * smoothness)
                    smooth_y = int(prev_y + (mouse_y - prev_y) * smoothness)
                    pyautogui.moveTo(smooth_x, smooth_y)
                    prev_x, prev_y = smooth_x, smooth_y

                # Scroll: index + middle fingers raised.
                scroll_mode = (
                    index_up and middle_up and
                    not ring_up and not pinky_up
                )
                if scroll_mode:
                    current_y = (
                        hand_landmarks[8].y + hand_landmarks[12].y
                    ) / 2
                    if prev_scroll_y is not None:
                        movement = prev_scroll_y - current_y
                        if movement > 0.02:
                            pyautogui.scroll(30)
                        elif movement < -0.02:
                            pyautogui.scroll(-30)
                    prev_scroll_y = current_y
                else:
                    prev_scroll_y = None
            else:
                # Do not run mouse actions while paused.
                status_text = "PAUSED - hold open palm to resume"

            if not paused:
                status_text = "ACTIVE - open palm to pause"

        else:
            pause_gesture_start = None
            prev_scroll_y = None
            status_text = "PAUSED - show open palm to resume" if paused else "ACTIVE - show hand"

        cv2.putText(
            frame, status_text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 255) if paused else (0, 255, 0), 2
        )
        cv2.imshow("Hand Gesture Controlled Computer", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
