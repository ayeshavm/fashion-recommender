import streamlit as st
import numpy as np
import pandas as pd
import random
import os
import pickle
import torch
from torchvision import transforms
from PIL import Image
import clip
import sys
sys.path.append("src")
from retrieval import normalize, fusion_retrieve, faiss_retrieve, clip_fusion_retrieve
from evaluation import log_retrieval_results
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- Config ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

category_dict = {
    "shirt": "top", "blouse": "top", "sweater":"top", "cardigan":"top", "vest":"top", "top":"top",
    "pant": "bottom", "skirt": "bottom", "jean": "bottom", "shorts": "bottom", "skort": "bottom", "trousers":"bottom",
    "dress": "dress", "jumpsuit": "jumpsuit",
    "jacket":"jacket", "blazer":"jacket", 
    "shoe": "shoes", "sneaker": "shoes", "boot": "shoes", "loafer": "shoes", "heels": "shoes","flats":"shoes",
    "sandals":"shoes",
    "bag": "accessory", "scarf": "accessory", "hat": "accessory"
}

complementary_rules = {
    "top": ["bottom", "shoes", "accessory", "jacket"],
    "bottom": ["top", "shoes", "accessory"],
    "dress": ["shoes", "accessory", "jacket"],
    "jumpsuit": ["shoes", "accessory", "jacket"],
    "shoes": ["top", "bottom", "accessory"],
    "accessory": ["top", "bottom", "shoes"]
}


# ---- Helper functions -----
# Load BLIP model
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption_from_image(pil_image):
    inputs = blip_processor(images=pil_image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def lookup_caption_from_filename(selected_image):
    return captions_df[captions_df["image"] == selected_image]["caption"].values[0]
    
def detect_item_type(image_caption):
    image_caption = image_caption.lower()
    for keyword, category in category_dict.items():
        if keyword in image_caption:
            return category
    return "unknown"

def get_complementary_items(detected_type, gallery_type_map):
    suggestions = []
    targets = complementary_rules.get(detected_type, [])
    for target in targets:
        matching_files = [filename for filename, type_ in gallery_type_map.items() if type_ == target]
        if matching_files:
            picked = random.choice(matching_files)
            suggestions.append(picked)
    return suggestions

def get_similar_items(uploaded_image):
    # Placeholder for FAISS/CLIP similarity search
    # Return a list of image paths for similar items
    return []  

def get_results(mode, query_clip, query_gcn):
    k_plus = top_k + 1
    if mode == "CLIP+GCN Fusion":
        return fusion_retrieve(query_clip, query_gcn, clip_norm, gcn_norm, item_ids, k=k_plus, alpha=alpha)
    elif mode == "CLIP Cos+Euc":
        return clip_fusion_retrieve(query_clip, clip_norm, item_ids, k=k_plus, alpha=alpha)
    elif mode == "FAISS":
        return faiss_retrieve(query_clip.reshape(1, -1), clip_norm, item_ids, k=k_plus)
    else:
        return []

def remove_anchor(results, query_idx):
    if query_idx is None:
        return results
    return [(i, score) for i, score in results if i != query_idx]

# --- Load Embeddings + Metadata ---
output_dir = '/Users/ayeshamendoza/repos/fashion-recommender/data/output'
image_dir = '/Users/ayeshamendoza/repos/fashion-recommender/data/images/zara'
gallery_path = '/Users/ayeshamendoza/repos/fashion-recommender/data/images/my_gallery'

captions_df = pd.read_csv(os.path.join(output_dir, "mvp_zara_blip_captions_updated.csv"))

# Create gallery_type_map
gallery_type_map = {}
for ix, row in captions_df.iterrows():
    gallery_type_map[row["image"]] = detect_item_type(row["caption"])

clip_df = pd.read_csv(os.path.join(output_dir, "mvp_clip_image_embeddings.csv")).apply(pd.to_numeric, errors="coerce").drop(columns='filename')    
clip_embeddings = clip_df.values.astype(np.float32)

graph_path = os.path.join(output_dir, "mvp_fashion_graph_data.pt")
data = torch.load(graph_path, weights_only=False)
item_node_indices = (data.node_type == 0).nonzero(as_tuple=True)[0].tolist()
gcn_all = torch.load(os.path.join(output_dir, "node_logits.pt")).detach().cpu()
gcn_embeddings = gcn_all[item_node_indices].numpy().astype(np.float32)

clip_norm = normalize(clip_embeddings)
gcn_norm = normalize(gcn_embeddings)

# Load item metadata
items_df = pd.read_csv(os.path.join(output_dir,"item_names.csv"))
item_names = list(items_df['filename'].values)

with open(os.path.join(output_dir, "item_to_idx.pkl"), "rb") as f:
    item_to_idx = pickle.load(f)
idx_to_item = {v: k for k, v in item_to_idx.items()}
item_ids = list(range(len(clip_norm)))



# --- UI: Image Upload / Gallery ---
st.title("AI Fashion Recommender 🌸 - Your Style Ally")
option = st.radio("How would you like to upload a clothing item?", ["Upload an image", "Choose from gallery"])

display_img = None
if option == "Upload an image":
    uploaded_file = st.file_uploader("Upload a clothing image", type=["jpg", "png", "jpeg"])
    # Get caption for uploaded file
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        uploaded_caption = generate_caption_from_image(image)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write(f"📝 Caption: {uploaded_caption}")
        display_img = image

elif option == "Choose from gallery":
    gallery_images = [f for f in os.listdir(gallery_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    st.write("### Select an item from your closet:")
    selected_image = None
    
    
    cols = st.columns(3)
    for idx, image_name in enumerate(gallery_images):
        img_path = os.path.join(gallery_path, image_name)
        img = Image.open(img_path)
        with cols[idx % 3]:
            if st.button(image_name, key=image_name):
                selected_image = image_name
                
            st.image(img, use_container_width=True)
    if selected_image:
        st.write(f"**You selected:** `{selected_image}`")
        display_img = Image.open(os.path.join(gallery_path, selected_image))
        st.image(display_img, caption="Selected Image", use_container_width=True)
        

# --- Unified Flow: If User Uploaded or Selected an Image ---
if display_img is not None:
    image_tensor = preprocess(display_img).unsqueeze(0).to(device)
    with torch.no_grad():
        query_clip = model.encode_image(image_tensor).cpu().numpy().squeeze()
    query_gcn = np.zeros_like(query_clip)

    st.subheader("🔍 Retrieval Based on Your Image")
    mode = st.selectbox("Choose Retrieval Mode", ["CLIP+GCN Fusion", "CLIP Cos+Euc", "FAISS"], key="user_query")
    top_k = st.slider("Number of Results", 1, 10, 5, key="top_k_user")
    alpha = st.slider("Fusion Blend (CLIP vs GCN)", 0.0, 1.0, 0.7, key="alpha_user")
    
    # prevent crash if GCN isn't available
    if option == "Upload an image" and mode == "CLIP+GCN Fusion":
        st.warning("GCN Fusion only works for indexed Zara items. Switching to CLIP Cos+Euc.")
        mode = "CLIP Cos+Euc"

    query_gcn = np.zeros((gcn_norm.shape[1],), dtype=np.float32)  # correct shape fallback
    
    # Remove anchor if gallery selection overlaps Zara items
    if option == "Choose from gallery" and selected_image:
        try:
            query_idx = item_names.index(selected_image)
        except ValueError:
            query_idx = None
    else:
        query_idx = None
        
    results = remove_anchor(get_results(mode, query_clip, query_gcn), query_idx)[:top_k]
    st.write("### 👗 Recommended Matches")
    for i, (idx, score) in enumerate(results):
        st.image(f"{image_dir}/{item_names[idx]}", caption=f"{item_names[idx]} (score: {score:.2f})", width=200)
        
    # --- Complete the Look ---
    st.markdown("---")
    st.subheader("🧵 Complete the Look:")

    # Detect item type (from filename)
    if option == "Upload an image" and uploaded_file is not None:
        detected_type = detect_item_type(uploaded_caption)
    elif option == "Choose from gallery" and selected_image is not None:
        selected_caption = lookup_caption_from_filename(selected_image)
        detected_type = detect_item_type(selected_caption)
    else:
        detected_type = "unknown"

    st.write(f"🔍 Detected Item Type: `{detected_type}`")

    # Suggest complementary items
    gallery_images = [f for f in os.listdir(gallery_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    complementary_items = get_complementary_items(detected_type, gallery_type_map)

    # Display complementary items
    if complementary_items:
        st.write("Here are some items to complete your look:")
        for comp_file in complementary_items:
            comp_img_path = os.path.join(gallery_path, comp_file)
            comp_img = Image.open(comp_img_path)
            st.image(comp_img, caption=comp_file, use_container_width=True)
    else:
        st.write("No complementary items found.")



else:
    # --- Default Flow: Index-Based Comparison ---
    st.title("Fashion Recommender: Fusion + FAISS")
    st.subheader("🎛️ Recommendation Mode Comparison")

    mode_a = st.selectbox("Choose First Mode", ["CLIP+GCN Fusion", "CLIP Cos+Euc", "FAISS"], key="mode_a")
    mode_b = st.selectbox("Choose Second Mode", ["CLIP+GCN Fusion", "CLIP Cos+Euc", "FAISS"], key="mode_b")
    query_idx = st.slider("Select a query item", 0, len(clip_norm)-1, 5)
    alpha = st.slider("Blend (CLIP vs GCN)", 0.0, 1.0, 0.7)
    top_k = st.slider("How many recommendations?", 1, 10, 5)

    query_clip = clip_norm[query_idx]
    query_gcn = gcn_norm[query_idx]

    results_a = remove_anchor(get_results(mode_a, query_clip, query_gcn), query_idx)[:top_k]
    results_b = remove_anchor(get_results(mode_b, query_clip, query_gcn), query_idx)[:top_k]

    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "retrieval_logs.csv")
    log_retrieval_results(query_idx, mode_a, results_a, alpha, filename=log_file)
    log_retrieval_results(query_idx, mode_b, results_b, alpha, filename=log_file)

    st.subheader("🎯 Anchor Item (Query)")
    st.image(f"{image_dir}/{item_names[query_idx]}", caption=f"Item #{query_idx}: {item_names[query_idx]}", use_container_width=True)
    st.markdown("These are the recommendations based on the item above.")
    st.markdown(f"### 🔍 Comparing: {mode_a} vs {mode_b}")

    for i in range(top_k):
        col1, col2 = st.columns(2)
        idx_a, score_a = results_a[i]
        idx_b, score_b = results_b[i]
        with col1:
            st.image(f"{image_dir}/{item_names[idx_a]}", caption=f"{mode_a}\nScore: {score_a:.4f}", use_container_width=True)
        with col2:
            st.image(f"{image_dir}/{item_names[idx_b]}", caption=f"{mode_b}\nScore: {score_b:.4f}", use_container_width=True)

    overlap = len(set(i for i, _ in results_a).intersection(i for i, _ in results_b))
    overlap_pct = overlap / top_k * 100
    st.markdown(f"**Top-{top_k} Overlap:** `{overlap}` items ({overlap_pct:.1f}%)")
