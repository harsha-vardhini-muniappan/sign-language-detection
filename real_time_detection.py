import cv2
import mediapipe as mp
import numpy as np
import joblib


model = joblib.load('sign_language_model.pkl')


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

def extract_hand_landmarks(hand_landmarks):
   
    data = []
    for lm in hand_landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z])
    return data

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

           
            features = extract_hand_landmarks(hand_landmarks)
            features_np = np.array(features).reshape(1, -1)

           
            prediction = model.predict(features_np)
            predicted_sign = prediction[0]

          
            cv2.putText(frame, f'Sign: {predicted_sign}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Sign Language Detection", frame)

  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
