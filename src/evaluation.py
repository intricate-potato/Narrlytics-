
from sklearn.metrics import roc_auc_score, ndcg_score
import numpy as np

class Evaluator:
    def __init__(self, config):
        self.config = config

    def calculate_auc(self, true_labels, pred_scores):
        return roc_auc_score(true_labels, pred_scores)

    def calculate_mrr(self, recommendations, ground_truth):
        mrr_scores = []
        for user, recs in recommendations.items():
            for i, rec in enumerate(recs):
                if rec in ground_truth.get(user, []):
                    mrr_scores.append(1 / (i + 1))
                    break
            else:
                mrr_scores.append(0)
        return np.mean(mrr_scores)

    def calculate_ndcg(self, recommendations, ground_truth, k):
        ndcg_scores = []
        for user, recs in recommendations.items():
            true_relevance = np.zeros(len(recs))
            for i, rec in enumerate(recs):
                if rec in ground_truth.get(user, []):
                    true_relevance[i] = 1
            
            if np.sum(true_relevance) > 0:
                ndcg_scores.append(ndcg_score([true_relevance], [np.arange(len(recs)) + 1], k=k))
            else:
                ndcg_scores.append(0)
        return np.mean(ndcg_scores)
