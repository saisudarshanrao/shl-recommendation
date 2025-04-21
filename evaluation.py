
import numpy as np
from sklearn.metrics import average_precision_score

def mean_recall_at_k(actual, predicted, k=3):
    recalls = []
    for a, p in zip(actual, predicted):
        relevant = set(a)
        retrieved = p[:k]
        hits = sum(1 for doc in retrieved if doc in relevant)
        recalls.append(hits / len(a))
    return np.mean(recalls)

def map_at_k(actual, predicted, k=3):
    ap_scores = []
    for a, p in zip(actual, predicted):
        relevant = np.array([1 if doc in a else 0 for doc in p[:k]])
        if sum(relevant) == 0:
            ap = 0.0
        else:
            precision = np.cumsum(relevant) / np.arange(1, k+1)
            ap = np.sum(precision * relevant) / min(len(a), k)
        ap_scores.append(ap)
    return np.mean(ap_scores)


ground_truth = [
    ["SHL Verify Interactive - Java", "SHL Team Collaboration Simulation"],
    ["SHL Python Coding Test", "SHL SQL Knowledge Assessment"]
]

predictions = [
    recommend_assessments("Java developer collaboration 40 mins"),
    recommend_assessments("Python SQL developer 60 mins")
]

print(f"Mean Recall@3: {mean_recall_at_k(ground_truth, predictions)}")
print(f"MAP@3: {map_at_k(ground_truth, predictions)}")
