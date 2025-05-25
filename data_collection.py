import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

label = input("Enter label (e.g., A, B, Hello): ")
cap = cv2.VideoCapture(0)
data = []

while True:
    success, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmark.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks.append(label)
            data.append(landmarks)

    cv2.imshow("Collecting", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or len(data) >= 200:
        break

cap.release()
cv2.destroyAllWindows()

with open("sign_data.csv", 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

print("Saved data for:", label)