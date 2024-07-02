from dotenv import load_dotenv
import os
load_dotenv()

from openai import OpenAI

# Set the API key and model name
MODEL = "gpt-4o"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from pdf2image import convert_from_path
import io
import base64

PDF_PATH = "test.pdf"
images = convert_from_path(PDF_PATH)

# Encode the images to base64 and save them as JPG files
def encode_image(image, index):
    jpg_path = f"page_{index}.jpg"
    image.save(jpg_path, format="JPEG")
    
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

base64_images = [encode_image(img, idx) for idx, img in enumerate(images)]

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Extract text from the provided images."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in base64_images
        ]}
    ],
    temperature=0.0,
)

# Save the extracted text to a file
with open("extracted_text.txt", "w") as text_file:
    for choice in response.choices:
        text_file.write(choice.message.content + "\n")

print("OCR process completed. Extracted text saved to 'extracted_text.txt'.")
