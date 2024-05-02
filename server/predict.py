import numpy as np
import cv2

def get_prediction(img, model):
    inv_img = np.invert(img)
    resized = cv2.resize(inv_img, (28, 28))
    flattened = resized / 255.0

    # Reshape the image to match the model's input shape
    input_data = np.expand_dims(flattened, axis=0)
    confidence_threshold = 0.9


    probabilities = model.predict(input_data)
    prediction = np.argmax(probabilities)
    confidence = probabilities[0][prediction]
    if confidence < confidence_threshold:
        return 0
    return np.argmax(probabilities)