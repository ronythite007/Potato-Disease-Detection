from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from io import BytesIO
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Configuration variables
MODEL_PATH = os.getenv("MODEL_PATH", "./model/potatoes.h5")
HTML_PATH = os.getenv("HTML_PATH", "index.html")
IMAGE_SIZE = (256, 256)
ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png"]

# Load the trained model
if not os.path.exists(MODEL_PATH):
    logger.error(f"Model file not found at path: {MODEL_PATH}")
    exit()

logger.info("Loading model...")
loaded_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
loaded_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
logger.info("Model loaded successfully")

# Class names for prediction
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Routes
@app.get("/", response_class=HTMLResponse)
async def serve_html():
    if not os.path.exists(HTML_PATH):
        return HTMLResponse("<h1>Frontend HTML file not found!</h1>", status_code=404)
    with open(HTML_PATH, "r") as file:
        return file.read()


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        return JSONResponse(
            {"error": f"Invalid file type. Only JPEG and PNG are supported. Received: {file.content_type}"},
            status_code=400,
        )
    
    try:
        # Read the image file as a byte stream
        image_bytes = await file.read()

        # Convert the byte stream to a PIL Image
        img = Image.open(BytesIO(image_bytes))
        logger.info("Image opened successfully")
        
        # Ensure the image is in RGB mode
        img = img.convert("RGB")
        logger.info("Image converted to RGB")

        # Resize to the target image size
        img = img.resize(IMAGE_SIZE)
        logger.info(f"Image resized to {IMAGE_SIZE}")

        # Convert the image to an array and normalize it
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        logger.info("Image array prepared")

        # Make prediction
        predictions = loaded_model.predict(img_array)
        logger.info(f"Prediction made: {predictions}")

        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        # Return the predicted class and confidence as a JSON response
        return JSONResponse({
            "predicted_class": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        logger.error(f"Error occurred during prediction: {e}")
        return JSONResponse(
            {"error": f"Failed to process image or make prediction: {str(e)}"},
            status_code=500,
        )

# uvicorn main:app --reload  use this command to run
