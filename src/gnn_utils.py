# Vision-Language Annotation (BLIP)
# Generate image captions using BLIP  
#    - Model: `Salesforce/blip-image-captioning-base`  
#    - Input: `zara/` images  
#    - Output: `zara_captions.csv` with columns:  
#      - `image`: filename  
#      - `caption`: generated text
     
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from pathlib import Path
import pandas as pd
import os

# Load BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image_dir = Path('drive/MyDrive/2025/graph-fashion-recsys/data/images/zara')
output_dir = 'drive/MyDrive/2025/graph-fashion-recsys/data/output'
image_paths = sorted(image_dir.glob("*.jpg"))

def generate_blip_captions(img_path):
    
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print('caption', caption, caption)

    return caption