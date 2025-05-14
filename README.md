# AI Fashion Recommender – Your Personal Styling Assistant

AI Fashion Recommender is an AI-powered fashion app that helps you complete your outfit with style. Upload a clothing item or choose from your closet gallery, and the app will suggest complementary pieces and similar items to create a complete look.

## 🧠 How It Works (Technical Overview)
The system combines image captioning, embedding-based retrieval, and rule-based logic to generate outfit suggestions:

- Uploaded or gallery images are processed with BLIP to generate natural language captions
- These captions are mapped to general item types (e.g., “top”, “shoes”) using simple tag-style parsing
- CLIP image embeddings power a visual similarity search, retrieving items with similar styles
- A rule-based logic layer then recommends complementary items based on the detected item type (e.g., matching tops with bottoms)

All components run in a Streamlit UI, designed for both desktop and mobile access.

## ✨ Features

- ✅ **Closet Camera Upload:** Upload your own clothing item image
- ✅ **Gallery Selection:** Browse and select from a gallery of your existing items
- ✅ **Image Captioning with BLIP:** Generates natural language descriptions of uploaded or gallery images
- ✅ **Item Type Detection:** Maps image captions to general clothing categories
- ✅ **Complete-the-Look Recommendations:** Suggests complementary items based on detected type using rule-based logic
- ✅ **Visual Similarity Recommendations:** Retrieve similar items using CLIP image embeddings
- ✅ **Mobile-Friendly UI:** Works across desktop and mobile screens

## 🖼️ **Example Flow**

1. Upload an image of your clothing item **or** select from gallery.
![Upload Flow](data/screenshots/select_gallery_img.png)
2. Generate caption (powered by BLIP) for item type identification.
3. View detected item type (e.g. “top,” “shoes”).
4. Get recommendations for similar items.
![Selected Item](data/screenshots/item_gallery_img.png)
![Item Recommendations](data/screenshots/reco_gallery_img.png)
5. View recommendations of complementary items to complete the look.
![Complete the Look](data/screenshots/ctl_gallery_img.png)

## 🚀 **Technologies Used**

- Streamlit (UI)
- CLIP (image embeddings for similarity)
- BLIP (image captioning)
- PyTorch
- FAISS (planned integration)
- Python

## 📂 **Project Structure**
```
/data/
/images/ → input images (gallery)
/output/ → saved captions, embeddings
/ src /
   retrieval.py → retrieval functions
   evaluation.py → logging & evaluation utils
app.py → main Streamlit app
```

## 📝 **Known Limitations**

- Caption accuracy may vary 
- Rule-based recommendation logic → may miss nuanced styling contexts
- Gallery UX in progress (currently shows all images every time)
