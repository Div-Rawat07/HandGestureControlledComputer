import cv2
import mediapipe as mp
import pyautogui

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import math


screen_width, screen_height = pyautogui.size()

print("Screen:", screen_width, screen_height)



def is_click(hand_landmarks):
    thumb = hand_landmarks[4]
    index = hand_landmarks[8]

    distance = math.sqrt(
        (thumb.x - index.x) ** 2 +
        (thumb.y - index.y) ** 2
    )

    return distance < 0.05

def is_right_click(hand_landmarks):
    thumb = hand_landmarks[4]
    middle = hand_landmarks[12]

    distance = math.sqrt(
        (thumb.x - middle.x) ** 2 +
        (thumb.y - middle.y) ** 2
    )

    return distance < 0.05

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


# Path to the hand landmark model
MODEL_PATH = "hand_landmarker.task"


# Create the MediaPipe hand landmarker
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)


# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

prev_x = 0
prev_y = 0

smoothness = 0.2
clicking = False
right_clicking = False
prev_scroll_y = None


while True:

    # Read frame
    success, frame = cap.read()

    if not success:
        print(" Failed to access webcam...")
        break


    # OpenCV uses BGR
    # MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Convert OpenCV image to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # Detect hands
    results = detector.detect(mp_image)


    # Check if hands were detected
    if results.hand_landmarks:

        for hand_landmarks in results.hand_landmarks:

            # Draw every landmark
            for landmark in hand_landmarks:

                h, w, _ = frame.shape

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Index fingertip = landmark 8
            index_tip = hand_landmarks[8]
            index_up = is_finger_up(hand_landmarks, 8, 6)
            middle_up = is_finger_up(hand_landmarks, 12, 10)
            ring_up = is_finger_up(hand_landmarks, 16, 14)
            pinky_up = is_finger_up(hand_landmarks, 20, 18)
            
            thumb_up = is_thumb_up(hand_landmarks)

            click = is_click(hand_landmarks)
            if click and not clicking:
                pyautogui.click()
                clicking = True


            elif not click:
                clicking = False

            # print("Click:", click)

            right_click = is_right_click(hand_landmarks)

            if right_click and not right_clicking:
                pyautogui.rightClick()
                right_clicking = True

            elif not right_click:
                right_clicking = False

            # print("Right Click:", right_click)

            # Move mouse only when pointing
            if index_up and not middle_up and not ring_up and not pinky_up:

                mouse_x = int(index_tip.x * screen_width)
                mouse_y = int(index_tip.y * screen_height)

                smooth_x = int(prev_x + (mouse_x - prev_x) * smoothness)
                smooth_y = int(prev_y + (mouse_y - prev_y) * smoothness)

                pyautogui.moveTo(smooth_x, smooth_y)

                prev_x = smooth_x
                prev_y = smooth_y

            scroll_mode = index_up and middle_up and not ring_up and not pinky_up

            # print("Scroll Mode:", scroll_mode)

            if scroll_mode:

                current_y = (hand_landmarks[8].y + hand_landmarks[12].y) / 2

                if prev_scroll_y is not None:

                    movement = prev_scroll_y - current_y

                    if movement > 0.02:
                        pyautogui.scroll(30)

                    elif movement < -0.02:
                        pyautogui.scroll(-30)

                prev_scroll_y = current_y

            else:
                prev_scroll_y = None

            # Count fingers
            finger_count = 0

            if index_up:
                finger_count += 1

            if middle_up:
                finger_count += 1

            if ring_up:
                finger_count += 1

            if pinky_up:
                finger_count += 1

            if thumb_up:
                finger_count += 1

            # print("Thumb:", thumb_up)
            # print("Index finger:", index_up)
            # print("Middle finger:", middle_up)
            # print("Ring finger:", ring_up)
            # print("Pinky finger:", pinky_up)
            # print("Finger count:", finger_count)

            gesture = detect_gesture(
                thumb_up,
                index_up,
                middle_up,
                ring_up,
                pinky_up
            )

            # print("Gesture:", gesture)

            # print(
            #     "Index Tip:",
            #     index_tip.x,
            #     index_tip.y,
            #     index_tip.z
            # )

                
    # Show webcam
    cv2.imshow(
        "Hand Detection",
        frame
    )


    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Cleanup
cap.release()
cv2.destroyAllWindows()