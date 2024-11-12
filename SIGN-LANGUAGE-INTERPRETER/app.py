import cv2
import numpy as np
from keras.models import load_model
import src.configs as cf

# Load the pre-trained model
model = load_model('./model/fine_tune_asl_model.h5')
# model = load_model('./model/cnn_asl_model.h5')

def recognize():
    # Initialize camera with index 1 (change to 0 if you only have one camera)
    cam = cv2.VideoCapture(0)  # Start with index 0
    if not cam.isOpened():
        print("Error: Could not open camera.")
        return

    text = ""
    word = ""
    count_same_frame = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break
		
        # Target ar	ea where the hand gestures should be
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (0, 0), (cf.CROP_SIZE, cf.CROP_SIZE), (0, 255, 0), 3)
        
        # Preprocess the frame for the model
        cropped_image = frame[0:cf.CROP_SIZE, 0:cf.CROP_SIZE]
        resized_frame = cv2.resize(cropped_image, (cf.IMAGE_SIZE, cf.IMAGE_SIZE))
        reshaped_frame = np.array(resized_frame).reshape((1, cf.IMAGE_SIZE, cf.IMAGE_SIZE, 3))
        frame_for_model = reshaped_frame / 255.0

        # Predict gesture
        old_text = text
        prediction = model.predict(frame_for_model)
        prediction_probability = prediction[0, prediction.argmax()]
        text = cf.CLASSES[prediction.argmax()]  # Selecting the max confidence index
        
        # Handle special cases for the text
        if text == 'space':
            text = '_'
        if text != 'nothing':
            if old_text == text:
                count_same_frame += 1
            else:
                count_same_frame = 0

            if count_same_frame > 10:
                word += text
                count_same_frame = 0

        # Display prediction and probability
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blackboard, " ", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0, 0))
        cv2.putText(blackboard, f"Predict: {text}", (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
        cv2.putText(blackboard, f"Probability: {prediction_probability * 100:.2f}%", (30, 170), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
        cv2.putText(blackboard, word, (30, 300), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
        
        # Combine the original frame and the blackboard for display
        res = np.hstack((frame, blackboard))
        cv2.imshow("Recognizing gesture", res)
        
        # Handle key presses
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):  # Quit
            break
        if k == ord('r'):  # Reset word
            word = ""
        if k == ord('z'):  # Remove last character
            word = word[:-1]

    # Release resources
    cam.release()
    cv2.destroyAllWindows()

# Run the recognition function
recognize()
