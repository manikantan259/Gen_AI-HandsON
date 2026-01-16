import os
from google import genai
from PIL import Image


client = genai.Client(
    api_key="AIzaSyACDGYykEj-kaN7BlJsGKOuaolwXzoXbmI"   
)



img_path = r"C:\Users\rajku\OneDrive\Desktop\Assignement\Week_1\HandsOn\img.jpg"
image = Image.open(img_path)


response = client.models.generate_content(
    model="models/gemini-2.5-flash-image",
    contents=[
        "Extract all text and convert invoice to structured JSON with meaningful tags.",
        image
    ]
)

print("=== Extracted Output ===\n")
print(response.text)
