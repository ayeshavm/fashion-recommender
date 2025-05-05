import pandas as pd
from datetime import datetime

def log_retrieval_results(query_idx, mode, results, alpha=None, filename="retrieval_logs.csv"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    records = []

    for rank, (item_idx, score) in enumerate(results):
        records.append({
            "timestamp": now,
            "query_idx": query_idx,
            "mode": mode,
            "rank": rank + 1,
            "item_idx": item_idx,
            "score": score,
            "alpha": alpha
        })

    df = pd.DataFrame(records)

    try:
        existing = pd.read_csv(filename)
        df = pd.concat([existing, df], ignore_index=True)
    except FileNotFoundError:
        pass

    df.to_csv(filename, index=False)
