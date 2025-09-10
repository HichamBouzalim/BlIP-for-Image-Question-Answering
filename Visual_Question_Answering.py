# Import necessary libraries
import requests  # For downloading images from a URL
from PIL import Image  # For image loading and processing
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering  # BLIP model and processor for image captioning / VQA

# -----------------------------
# 1. Load the BLIP processor and model
# -----------------------------
# The processor prepares the image and question for the VQA model
processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

# The model is trained for Visual Question Answering tasks
# It can answer questions about objects, actions, locations, and yes/no questions
model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# -----------------------------
# 2. Load the image
# -----------------------------
# Define the image URL (can be any accessible online image)
img_url = 'https://github.com/HichamBouzalim/BLIP-for-image-captioning/blob/main/Picture.jpg?raw=true'

# Open the image from the URL
# 'stream=True' ensures the image is downloaded in chunks (efficient for large files)
# Convert the image to RGB mode, which is the standard format expected by the model
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# -----------------------------
# 3. Define the questions
# -----------------------------
# You can ask multiple questions about the image
# These questions are passed along with the image to the model for generating answers
questions = [
    "Is there an animal in the image?",  # Describes actions or events in the image
    "What is happening in the image?"             # Identifies main objects, people, or animals
]

# -----------------------------
# 4. Generate answers for each question
# -----------------------------
for question in questions:
    # Prepare the inputs for the model: combine image and question into tensors
    inputs = processor(raw_image, question, return_tensors="pt")
    
    # Generate the answer using the BLIP model
    # The model outputs token IDs which need to be decoded to human-readable text
    out = model.generate(**inputs)
    
    # Decode the generated token IDs into text and remove special tokens
    answer = processor.decode(out[0], skip_special_tokens=True)
    
    # Print the question and corresponding answer
    print(f"Question: {question}")
    print(f"Answer: {answer}\n")
