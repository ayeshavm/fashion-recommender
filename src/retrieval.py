import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import faiss

def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def fusion_retrieve(query_clip, query_gcn, item_clip, item_gcn, item_ids, k=5, alpha=0.7):
    clip_sims = cosine_similarity(query_clip.reshape(1, -1), item_clip)[0]
    gcn_sims  = cosine_similarity(query_gcn.reshape(1, -1), item_gcn)[0]
    fusion_scores = alpha * clip_sims + (1 - alpha) * gcn_sims
    top_k_indices = fusion_scores.argsort()[-k:][::-1]
    return [(item_ids[i], fusion_scores[i]) for i in top_k_indices]

def faiss_retrieve(query_embedding, item_embeddings, item_ids, k=5):
    dim = item_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine similarity when normalized
    index.add(item_embeddings.astype('float32'))
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    distances, indices = index.search(query_embedding, k)
    return [(item_ids[i], distances[0][j]) for j, i in enumerate(indices[0])]

def clip_fusion_retrieve(query_clip, item_clip, item_ids, k=5, alpha=0.5):
    """
    Retrieves top-k items by combining cosine and euclidean similarities.
    alpha: weight for cosine similarity (0.0 to 1.0)
    """
    # Cosine similarity (higher is better)
    cosine_sim = cosine_similarity(query_clip.reshape(1, -1), item_clip)[0]

    # Euclidean distance (convert to similarity)
    euclidean_dist = euclidean_distances(query_clip.reshape(1, -1), item_clip)[0]
    euclidean_sim = 1 / (1 + euclidean_dist)  # higher is better

    # Fusion score
    fusion_score = alpha * cosine_sim + (1 - alpha) * euclidean_sim
    top_k_indices = fusion_score.argsort()[-k:][::-1]

    return [(item_ids[i], fusion_score[i]) for i in top_k_indices]
