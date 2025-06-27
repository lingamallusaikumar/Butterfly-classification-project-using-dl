# app.py
# This is the main Python file for our Flask web application.

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

# 1. INITIALIZE THE FLASK APP
# ---------------------------------
app = Flask(__name__)

# 2. LOAD THE H5 MODEL
# ---------------------------------
# Define the path to your .h5 model file.
MODEL_PATH = 'models//my_model.h5'

try:
    # Use TensorFlow's Keras module to load the pre-trained model.
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"⚠️ Could not load the model. Error: {e}. The app will run in dummy mode.")

# 3. CONFIGURE MODEL PARAMETERS
# ---------------------------------
# CRITICAL: Replace these with the actual class names your model was trained on.
# The order must be exactly the same as in your training data.
CLASS_NAMES = ['ADONIS', 'AFRICAN GIANT SWALLOWTAIL', 'AMERICAN SNOOT', 'AN 88', 'APPOLLO', 'ATALA', 'BANDED ORANGE HELICONIAN', 'BANDED PEACOCK', 'BECKERS WHITE', 'BLACK HAIRSTREAK', 'BLUE MORPHO', 'BLUE SPOTTED CROW', 'BROWN SIPROETA', 'CABBAGE WHITE', 'CAIRNS BIRDWING', 'CHECQUERED SKIPPER', 'CHESTNUT', 'CLEOPATRA', 'CLODIUS PARNASSIAN', 'CLOUDED SULPHUR', 'COMMON BANDED AWL', 'COMMON WOOD-NYMPH', 'COPPER TAIL', 'CRECENT', 'CRIMSON PATCH', 'DANAID EGGFLY', 'EASTERN COMA', 'EASTERN DAPPLE WHITE', 'EASTERN PINE ELFIN', 'ELBOWED PIERROT', 'GOLD BANDED', 'GREAT EGGFLY', 'GREAT JAY', 'GREEN CELLED CATTLEHEART', 'GREY HAIRSTREAK', 'INDRA SWALLOW', 'IPHICLUS SISTER', 'JULIA', 'LARGE MARBLE', 'MALACHITE', 'MANGROVE SKIPPER', 'MESTRA', 'METALMARK', 'MILBERTS TORTOISESHELL', 'MONARCH', 'MOURNING CLOAK', 'ORANGE OAKLEAF', 'ORANGE TIP', 'ORCHARD SWALLOW', 'PAINTED LADY', 'PAPER KITE', 'PEACOCK', 'PINE WHITE', 'PIPEVINE SWALLOW', 'POPINJAY', 'PURPLE HAIRSTREAK', 'PURPLISH COPPER', 'QUESTION MARK', 'RED ADMIRAL', 'RED CRACKER', 'RED POSTMAN', 'RED SPOTTED PURPLE', 'SCARCE SWALLOW', 'SILVER SPOT SKIPPER', 'SLEEPY ORANGE', 'SOOTYWING', 'SOUTHERN DOGFACE', 'STRAITED QUEEN', 'TROPICAL LEAFWING', 'TWO BARRED FLASHER', 'ULYSES', 'VICEROY', 'WOOD SATYR', 'YELLOW SWALLOW TAIL', 'ZEBRA LONG WING']

# CRITICAL: Change this to the image size your model expects (e.g., (150, 150)).
IMAGE_TARGET_SIZE = (224, 224)


# 4. DEFINE HELPER FUNCTIONS
# ---------------------------------
def preprocess_image(image_file):
    """
    Takes an uploaded image file, resizes it to the model's expected input size,
    and formats it as a NumPy array ready for prediction.
    """
    # Open the image using Pillow (PIL)
    img = Image.open(image_file.stream).convert('RGB')
    
    # Resize the image to the target size
    img = img.resize(IMAGE_TARGET_SIZE)
    
    # Convert the image to a NumPy array and normalize pixel values to be between 0 and 1
    img_array = np.array(img) / 255.0
    
    # Add a "batch" dimension to the array (from (224, 224, 3) to (1, 224, 224, 3))
    # The model expects a batch of images, even if we're only predicting one.
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# 5. DEFINE FLASK ROUTES (THE WEB PAGES/ENDPOINTS)
# ---------------------------------
@app.route('/', methods=['GET'])
def home():
    """
    This is the main route. It will find and render the 'code.html' file
    from the 'templates' folder when a user visits the website.
    """
    return render_template('code.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    This is the prediction endpoint. The HTML form sends the image here.
    This route will process the image, use the model to make a prediction,
    and return the result as JSON.
    """
    # Check if a file was sent in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    # Check if the user selected a file
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and model:
        try:
            # Preprocess the image to prepare it for the model
            processed_image = preprocess_image(file)
            
            # Use the loaded model to make a prediction
            predictions = model.predict(processed_image)
            
            # Process the prediction result
            predicted_index = np.argmax(predictions[0])
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(np.max(predictions[0]))
            
            # Return the result in a clean JSON format
            return jsonify({
                'prediction': predicted_class,
                'confidence': confidence
            })
        except Exception as e:
            print(f"Prediction Error: {e}")
            return jsonify({'error': 'Failed to process the image.'}), 500
            
    elif not model:
        # If the model failed to load, return an error
        return jsonify({'error': 'Model is not loaded. Please check the server logs.'}), 500
        
    return jsonify({'error': 'An unknown error occurred.'}), 500


# 6. RUN THE FLASK APP
# ---------------------------------
if __name__ == '__main__':
    # 'debug=True' is useful for development as it automatically reloads
    # the server when you make changes and shows detailed errors.
    app.run(debug=True)