import streamlit as st
import numpy as np
import pandas as pd
import  os
import pickle

import torch
from torchvision import transforms
import sys
sys.path.append("notebooks") 
from retrieval import normalize, fusion_retrieve, faiss_retrieve, clip_fusion_retrieve
from evaluation import log_retrieval_results

import clip  # make sure clip is installed (pip install git+https://github.com/openai/CLIP.git)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# --- Load Embeddings + Metadata ---
output_dir = '/Users/ayeshamendoza/repos/fashion-recommender/data/output'
image_dir = '/Users/ayeshamendoza/repos/fashion-recommender/data/images/zara'
gallery_path = '/Users/ayeshamendoza/repos/fashion-recommender/data/images/my_gallery'

## Prompt for input
from PIL import Image

st.title("Closet Camera + Style Recommender")

option = st.radio("How would you like to upload a clothing item?", ["Upload an image", "Choose from gallery"])

if option == "Upload an image":
    uploaded_file = st.file_uploader("Upload a clothing image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        display_img = image
        # TODO: Pass to similarity/complement recommender

elif option == "Choose from gallery":
    gallery_images = [f for f in os.listdir(gallery_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    st.write("### Select an item from your closet:")
    selected_image = None

    num_columns = 3
    cols = st.columns(num_columns)

    for idx, image_name in enumerate(gallery_images):
        image_path = os.path.join(gallery_path, image_name)
        img = Image.open(image_path)

        with cols[idx % num_columns]:
            if st.button(image_name, key=image_name):  # selection by filename
                selected_image = image_name
            st.image(img, use_container_width=True)

    if selected_image:
        st.write(f"**You selected:** `{selected_image}`")
        display_img = Image.open(os.path.join(gallery_path, selected_image))
        st.image(display_img, caption="Selected Image", use_container_width=True)
        # TODO: Pass to similarity/complement recommender

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])
if display_img is not None:
    image_tensor = preprocess(display_img).unsqueeze(0)  # shape: [1, 3, 224, 224]
# End of image input selection

clip = pd.read_csv(os.path.join(output_dir, "mvp_clip_image_embeddings.csv")).apply(pd.to_numeric, errors="coerce").drop(columns='filename')
clip_embeddings = clip.values.astype(np.float32)

graph_path = os.path.join(output_dir,"mvp_fashion_graph_data.pt")
data = torch.load(graph_path, weights_only=False)
item_node_indices = (data.node_type == 0).nonzero(as_tuple=True)[0]
item_node_indices = item_node_indices.tolist()

gcn_all = torch.load(os.path.join(output_dir, "node_logits.pt")).detach().cpu()
gcn_embeddings = gcn_all[item_node_indices].numpy().astype(np.float32)

# Normalize
clip_norm = normalize(clip_embeddings)
gcn_norm = normalize(gcn_embeddings)

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
    return [(i, score) for i, score in results if i != query_idx]

# Load item metadata (optional)
# Load the item to index mapping
items_df = pd.read_csv(os.path.join(output_dir,"item_names.csv"))
item_names = list(items_df['filename'].values)

with open(os.path.join(output_dir, "item_to_idx.pkl"), "rb") as f:
    item_to_idx = pickle.load(f)

# Reverse it: idx → item
idx_to_item = {v: k for k, v in item_to_idx.items()}

# --- UI ---
st.title("Fashion Recommender: Fusion + FAISS")

st.subheader("🎛️ Recommendation Mode Comparison")

mode_a = st.selectbox("Choose First Mode", ["CLIP+GCN Fusion", "CLIP Cos+Euc", "FAISS"], key="mode_a")
mode_b = st.selectbox("Choose Second Mode", ["CLIP+GCN Fusion", "CLIP Cos+Euc", "FAISS"], key="mode_b")

item_ids = list(range(len(clip_norm)))

query_idx = st.slider("Select a query item", 0, len(clip_norm)-1, 5)
alpha = st.slider("Blend (CLIP vs GCN)", 0.0, 1.0, 0.7)
top_k = st.slider("How many recommendations?", 1, 10, 5)

query_clip = clip_norm[query_idx]
query_gcn = gcn_norm[query_idx]

item_ids = list(range(len(clip_norm)))
# # --- Fusion Retrieval ---
results_a = get_results(mode_a, query_clip, query_gcn)

# # --- FAISS Retrieval ---
results_b = get_results(mode_b, query_clip, query_gcn)

# Remove anchor from results
results_a = remove_anchor(get_results(mode_a, query_clip, query_gcn), query_idx)[:top_k]
results_b = remove_anchor(get_results(mode_b, query_clip, query_gcn), query_idx)[:top_k]

# log results for evaluation
logs_dir = os.path.join(output_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, "retrieval_logs.csv")

log_retrieval_results(query_idx, mode_a, results_a, alpha, filename=log_file)
log_retrieval_results(query_idx, mode_b, results_b, alpha, filename=log_file)

# Display Anchor item image
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

indices_a = set([i for i, _ in results_a])
indices_b = set([i for i, _ in results_b])
overlap_count = len(indices_a.intersection(indices_b))
overlap_percent = overlap_count / top_k * 100

st.markdown(f"**Top-{top_k} Overlap:** `{overlap_count}` items ({overlap_percent:.1f}%)")