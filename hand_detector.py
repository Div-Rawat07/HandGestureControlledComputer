import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def is_finger_up(hand_landmarks, tip, pip):
    return hand_landmarks[tip].y < hand_landmarks[pip].y

def is_thumb_up(hand_landmarks):
    return hand_landmarks[4].x > hand_landmarks[3].x


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


while True:

    # Read frame
    success, frame = cap.read()

    if not success:
        print("❌ Failed to access webcam")
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

            print("Thumb:", thumb_up)
            print("Index finger:", index_up)
            print("Middle finger:", middle_up)
            print("Ring finger:", ring_up)
            print("Pinky finger:", pinky_up)
            print("Finger count:", finger_count)


            print(
                "Index Tip:",
                index_tip.x,
                index_tip.y,
                index_tip.z
            )

                


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