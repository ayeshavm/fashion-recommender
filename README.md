# AI Fashion Recommender – Your Personal Styling Assistant
AI Fashion Recommender is an AI-powered fashion app that helps you complete your outfit with style. Upload a clothing item or choose from your closet gallery, and the app will suggest complementary pieces and similar items to create a complete look.

## Introduction
Fashion Recommenders are behind nearly every shopping experience - from suggesting similar styles to curating entire outfits.  Whether you are browsing a product page or scrolling through a feed, recommendation systems quietly shape what we see next.

In this project, I built a Graph Neural Network (GNN) used for unsupervised fashion recommendation.  The model learns item relationships from co-occurrence data and visual context, allowing it to generate meaningful product suggestions based on learned embeddings.

## 🧩 **Problem Description**
Not everyone has an intuitive sense of style.  Choosing what to wear or even figuring out what goes together can often feel frustrating and time-consuming.  For many, shopping can be overwhelming, with endless options but little guidance tailored to their personal taste.  My goal is to reduce
decision fatigue and offer smart, contextual outfit suggestions based on visual and attribute relationships.

This project aims to ease that process by learning patterns in product relationships, and visual aesthetics.  By building a Graph Neural Network (GNN) model that captures the connections between fashion items, we can generate personalized and visually coherent product recommendations. Instead of relying on hard-coded rules or manual tagging, the system uses a Graph Neural Network (GNN) to learn embeddings that reflect nuanced relationships in fashion - from visual similarity to contextual co-occurrence.

These embeddings are then used to suggest similar or complementary items, helping users discover styles that feel both authentic and wearable.

## 🗂 **Data**
A curated collection of Zara product images, organized by clothing type and visual style:
- 52 images from zara.com were downloaded, some of these images are items available in my closet that can be used for testing recommendation.


## 🧹 **Data Processing**
- Renamed images into a standard naming convention.
- BLIP captions generated were not producing accurate captions, and have to be cleaned up for this MVP

Example of BLIP captions generated: 

`🖼️ zara_05.jpg → a white plea skirt with a black waistband` --> Updated to `a mini white pleated skirt with a black waistband`
![White pleated skirt](data/screenshots/zara_05.jpg)
`🖼️ zara_06.jpg → a brown jacket with a pepo collar and a pepo pepo pepo pepo pepo` --> Updated to `a knit peplum brown jacket with a peterpan collar`
![Brown peplum jacket](data/screenshots/zara_06.jpg)

- Generate the graph architecture, where
   - `Nodes` : item node i.e. each item image will have an item node
               attribute node i.e. each keyword will correspond to an attribute node
   - `Edges` : item -> attribute edge, and we have item -> item for item similarity edges


## 🔍 **EDA**
- When we plot the images in 2D space using UMAP, we can see how similar items are clustered together.
   - items i.e. sweaters, hats, pants are clustered together
   - and images with models are clustered together

![2D Visualization of CLIP embeddings in Space](data/screenshots/mvp-clip-embedding-2d-viz.png)

- Sample output from finding similar items using Cosine Similarity
![Anchor item + Similar items](data/screenshots/mvp-similarity.png)
![Anchor item + Similar items](data/screenshots/mvp-similarity-graph.png)

- After running `Graph Convolution Network (GCN)`
   - GCN is a neural network architecture designed to work on a graph-structured data.  It learns `node embeddings` by aggregating information from neighbors in the graph.
   - The final node embeddings are vector representations of each item that encode visual style, and reflect how items relate to each other in the graph.  Below is a visualization
     of the embeddings generated by the GCN.  
   - The final `top-K` items use these `node embeddings` to compute similarity scores between items
   ![Node Embeddings after GCN](data/screenshots/mvp-embeddings-afterGCN.png)


## 🧠 Methodology: How the Recommender Works:
This project combines multimodal embedding techniques and a graph-based learning framework to generate both visually similar and complementary outfit recommendations.

### Step 1: Image Captioning and Item Typing
- Uploaded or gallery images are passed through the BLIP model to generate natural language captions.
- Captions are then parsed to extract general item types (e.g., "skirt", "jacket", "top") using rule-based keyword tagging.

### Step 2: Embedding Generation (CLIP)
- Each image is encoded into a dense vector representation using CLIP, capturing its visual style and context.
- These embeddings are used for both:
  - Visual similarity (nearest neighbor search)
  - Graph construction and downstream learning

### Step 3: Graph Construction
We build a heterogeneous graph with two node types:
- **Item Nodes**: Each product image
- **Attribute Nodes**: Tags parsed from captions (e.g., "pleated", "brown", "jacket")

Edges include:
- `item → attribute` (from parsed captions)
- `item → item` (if visually similar or co-occur)

### Step 4: Graph Learning (GCN)
A Graph Convolutional Network (GCN) is trained on this item-attribute graph:
- The GCN aggregates node features across the graph structure to produce refined node embeddings.
- These embeddings encode visual and contextual relationships learned from the graph structure.
- Output embeddings are used for similarity scoring and recommendation retrieval (top-K).

GCN details:
- 2-layer GCN using PyTorch Geometric
- Aggregation function: mean
- Trained in unsupervised fashion to smooth embeddings across neighbors)


## ✨ Features

- 🧺 **Closet Camera Upload:** Upload your own clothing item image
- 🖼️ **Gallery Selection:** Browse and select from a gallery of your items
- 🧠 **Image Captioning (BLIP):** Generates descriptions of clothing items
- 🏷️ **Item Type Detection:** Maps captions to general fashion categories
- 💫 **Visual Similarity Search:** Retrieve visually similar items using CLIP embeddings
- 👗 **Complete-the-Look Suggestions:** Recommends complementary items using rule-based logic
- 📱 **Mobile-Friendly UI:** Built with Streamlit for desktop and mobile use

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
│
├── data/ # Screenshots, sample input, outputs
│ └── screenshots/ # Captures from UI and EDA
│
├── images/ # Raw input clothing images
├── output/ # Captions, embeddings
│
├── src/ # Core Python code
│ ├── retrieval.py # Embedding similarity + Top-K item search
│ ├── evaluation.py # Logging, visualization, and analysis
│ └── graph_utils.py # GCN graph creation and PyG training
│
├── app.py # Streamlit application
└── README.md
```


## 🧾 **Summary**

**EDA insights**
- Visual Clustering: UMAP projections of CLIP embeddings reveal clear clusters by item type (e.g., sweaters, pants, hats), with visually similar products naturally grouped.
- Embedding Space Coherence: Even with limited training data, items worn by models or photographed under similar lighting often cluster together.
- GCN Impact: After training the GCN on the item-attribute graph, node embeddings reflect both visual similarity and attribute-based relationships, improving recommendations.

**Known Limitations**

- Caption accuracy may vary 
- Rule-based recommendation logic → may miss nuanced styling contexts
- Gallery UX in progress (currently shows all images every time)

## 🔮 **Future Improvements**

- Replace rule-based logic with learned compatibility models (e.g., outfit scoring with Siamese or Transformer-based architecture)
- Improve item captioning using fine-tuned BLIP or LLaVA
- Add search, filters to improve UX.
- Expand graph structure with user-item interactions (if available)
