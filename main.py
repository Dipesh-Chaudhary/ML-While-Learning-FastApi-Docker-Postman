# main.py
import os
import torch
import openai
import requests

from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile

from pydantic import BaseModel

from transformers import BlipProcessor
from transformers import BlipForConditionalGeneration

from PIL import Image

from dotenv import load_dotenv  # Add this line

# Load environment variables from .env file
load_dotenv()  # Add this line

# Initialize FastAPI
app = FastAPI()
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Image Captioning and Response Generation API. Use the /predict endpoint to upload an image."}
# Set your OpenAI API key (Free tier usage)
openai.api_key = os.getenv("OPENAI_API_KEY")   # Replace with your OpenAI API key

# Initialize Hugging Face BLIP model (for image captioning)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Define a Pydantic model for responses
class PredictionResponse(BaseModel):
    caption: str
    response: str

# Step 1: Image Processing (using BLIP model)
def generate_caption(image_path):
    # Open image file
    raw_image = Image.open(image_path).convert("RGB")

    # Preprocess and predict caption
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption

# Step 2: Retrieve relevant documents (for RAG)
def retrieve_documents(caption):
    # Let's pretend we have a document collection or use an API for retrieval
    # For simplicity, we're using Wikipedia API here to fetch related information
    url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&list=search&srsearch={caption}"
    response = requests.get(url).json()
    
    documents = []
    if "query" in response:
        for result in response["query"]["search"]:
            documents.append(result["snippet"])
    
    return documents

# Step 3: Use OpenAI GPT-3 for generating a detailed response
def generate_response(caption, documents):
    prompt = f"Caption: {caption}\n\nDocuments:\n" + "\n".join(documents) + "\n\nProvide a detailed answer based on the above information."
    
    response = openai.Completion.create(
        model="text-davinci-003",  # Use the GPT-3 model
        prompt=prompt,
        temperature=0.7,
        max_tokens=500,
    )
    
    return response.choices[0].text.strip()


# API endpoint for image processing and response generation
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Save the uploaded image
    image_path = f"temp_{file.filename}"
    with open(image_path, "wb") as f:
        f.write(await file.read())
    
    # Step 1: Generate caption from the image
    caption = generate_caption(image_path)

    # Step 2: Retrieve relevant documents
    documents = retrieve_documents(caption)

    # Step 3: Generate a detailed response using GPT-3
    response = generate_response(caption, documents)

    # Return the caption and the detailed response
    return PredictionResponse(caption=caption, response=response)

