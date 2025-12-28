import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf
import pickle
import os

print("Loading model and encoder...")

# Load model
model = tf.keras.models.load_model('isl_hand_model.h5')
print("Model loaded")

# Load encoder
with open('label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
print("Encoder loaded")

# Setup MediaPipe detector
print("Setting up hand detector...")
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)
print("Detector ready")

def extract_landmarks_from_frame(frame):
    """Extract hand landmarks from a frame"""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)
    
    if not result.hand_landmarks:
        return None
    
    hand = result.hand_landmarks[0]
    vector = []
    for lm in hand:
        vector.extend([lm.x, lm.y, lm.z])
    
    return vector

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    print("\n=== ISL Real-Time Recognition ===")
    print("Press 'q' to quit")
    print("Show hand signs in front of camera\n")
    
    prev_label = None
    label_count = 0
    confidence_threshold = 0.7
    stability_frames = 3
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Extract landmarks
        features = extract_landmarks_from_frame(frame)
        
        if features is not None:
            # Reshape and predict
            features = np.array(features).reshape(1, -1)
            pred = model.predict(features, verbose=0)
            confidence = np.max(pred)
            
            if confidence > confidence_threshold:
                current_label = encoder.inverse_transform([np.argmax(pred)])[0]
                
                # Stabilize predictions
                if current_label == prev_label:
                    label_count += 1
                else:
                    prev_label = current_label
                    label_count = 1
                
                if label_count >= stability_frames:
                    display_label = current_label
                    display_conf = confidence * 100
                    
                    # Draw prediction on frame
                    cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 0), -1)
                    cv2.putText(frame, f"Sign: {display_label}", 
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                                1.2, (0, 255, 0), 3)
                    cv2.putText(frame, f"Conf: {display_conf:.1f}%", 
                                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Low confidence", 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 165, 255), 2)
        else:
            cv2.putText(frame, "No hand detected", 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2)
            prev_label = None
            label_count = 0
        
        cv2.putText(frame, "Press 'q' to quit", 
                    (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 1)
        
        cv2.imshow("ISL Recognition - LSIS", frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam closed")

if __name__ == "__main__":
    main()