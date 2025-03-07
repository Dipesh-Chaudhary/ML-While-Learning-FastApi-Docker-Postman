# main.py
import os
import torch
import openai
import requests

from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile

from openai import OpenAI


from pydantic import BaseModel

from transformers import BlipProcessor

from transformers import AutoTokenizer
from transformers import GPTNeoForCausalLM 
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
# # Set your OpenAI API key (Free tier usage)
# openai.api_key = os.getenv("OPENAI_API_KEY")   # Replace with your OpenAI API key
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize GPT-Neo model (for text generation)
gpt_neo_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
gpt_neo_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

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
    print
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


# Step 3: Use GPT-Neo for generating a detailed response
def generate_response(caption, documents):
    prompt = f"Caption: {caption}\n\nDocuments:\n" + "\n".join(documents) + "\n\nProvide a detailed answer based on the above information."
    
    # Tokenize the input prompt
    inputs = gpt_neo_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate text using GPT-Neo
    outputs = gpt_neo_model.generate(
        inputs.input_ids,
        max_length=500,
        temperature=0.7,
        num_return_sequences=1,
    )
    
    # Decode the generated text
    response = gpt_neo_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

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

    # Step 3: Generate a detailed response using GPT-Neo
    response = generate_response(caption, documents)

    # Return the caption and the detailed response
    return PredictionResponse(caption=caption, response=response)
#It is used since i have to access it through kaggle
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
