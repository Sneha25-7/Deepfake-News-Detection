import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model("models/deepfake_model.h5")

def predict_image(path):

    # Load image
    img = tf.keras.preprocessing.image.load_img(
        path,
        target_size=(128, 128)
    )

    # Convert to array
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Normalize
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img, verbose=0)[0][0]

    # Print prediction value in terminal
    print("Prediction Value:", prediction)
    # Classification (Reverse Labels)
    if prediction >= 0.5:
        result = "DEEPFAKE IMAGE"
        confidence = prediction * 100
    else:
        result = "REAL IMAGE"
        confidence = (1 - prediction) * 100

    return result, round(confidence, 2)