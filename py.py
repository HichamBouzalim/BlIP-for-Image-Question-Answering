# Import necessary libraries
import requests  # For downloading images from the internet
from PIL import Image  # For image processing
from transformers import BlipProcessor, BlipForConditionalGeneration  # BLIP model for image captioning and VQA

# -----------------------------
# 1. Load the BLIP processor and model
# -----------------------------
# The processor prepares the image and question for the model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
# The model generates captions or answers questions about the image
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# -----------------------------
# 2. Load the image
# -----------------------------
img_url = 'https://github.com/HichamBouzalim/BLIP-for-image-captioning/blob/main/Picture.jpg?raw=true'
# Open the image from the URL and convert it to RGB (standard for models)
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# -----------------------------
# 3. Define the question
# -----------------------------
# You can ask any question about the image
# List of questions to ask
questions = [
    "what is happening in the image?",
    "What is in the image?"
]

# Loop through questions and generate answers
for question in questions:
    # Prepare inputs for the model
    inputs = processor(raw_image, question, return_tensors="pt")
    
    # Generate answer
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    
    print(f"Question: {question}")
    print(f"Answer: {answer}\n")