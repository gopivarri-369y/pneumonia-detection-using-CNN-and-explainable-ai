from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import time
import uuid

app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained.h5")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# Ensure the static folder exists
if not os.path.exists("static"):
    os.makedirs("static")

# Preprocess image function
def preprocess_image(image_path, target_size=(300, 300)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    return img, img_array

# Function to generate LIME explanation
def generate_lime_explanation(image_array):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_array,
        model.predict,
        top_labels=1,          # Binary classification
        hide_color=0,          # Hide color for occluded regions
        num_samples=1000       # Number of perturbed samples
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],  # Get the predicted class
        positive_only=True,         # Show only regions contributing to prediction
        num_features=5,             # Number of regions to highlight
        hide_rest=False            # Show the rest of the image
    )
    # Convert to uint8 and scale
    temp = (temp * 255).astype(np.uint8)
    masked_img = mark_boundaries(temp, mask)
    return masked_img

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Generate a unique filename for the uploaded image
    unique_filename = str(uuid.uuid4()) + ".jpg"
    file_path = os.path.join("static", unique_filename)

    try:
        file.save(file_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

    try:
        # Preprocess the image
        original_image, image_array = preprocess_image(file_path)
        print("Preprocessed image shape:", image_array.shape)

        # Make a prediction using the model
        prediction = model.predict(np.expand_dims(image_array, axis=0))
        predicted_class = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
        confidence = prediction[0][0] if predicted_class == "Pneumonia" else 1 - prediction[0][0]

        # Generate LIME explanation
        lime_explanation = generate_lime_explanation(image_array)

        # Save the LIME explanation image with a unique name
        lime_filename = str(uuid.uuid4()) + "_lime.jpg"
        lime_path = os.path.join("static", lime_filename)
        plt.imsave(lime_path, lime_explanation, format='jpg')
        print(f"LIME explanation saved to: {lime_path}")

        # Append a timestamp to avoid caching
        timestamp = int(time.time())
        return jsonify({
            "prediction": predicted_class,
            "confidence": float(confidence),
            "image_url": f"/static/{unique_filename}?{timestamp}",
            "lime_url": f"/static/{lime_filename}?{timestamp}"
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)