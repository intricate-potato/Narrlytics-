
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class NarrativeRecommender:
    def __init__(self, config, article_embeddings):
        self.config = config
        self.article_embeddings = article_embeddings

    def build_user_narrative_profile(self, user_history):
        if not user_history:
            return np.zeros(self.config['model']['reduced_dimension'] * 6)
        
        history_embeddings = [self.article_embeddings[news_id] for news_id in user_history if news_id in self.article_embeddings]
        
        if not history_embeddings:
            return np.zeros(self.config['model']['reduced_dimension'] * 6)
            
        return np.mean(history_embeddings, axis=0)

    def recommend_narrative_only(self, user_id, user_history, candidates, k):
        user_profile = self.build_user_narrative_profile(user_history)
        candidate_embeddings = np.array([self.article_embeddings[c] for c in candidates if c in self.article_embeddings])
        
        if candidate_embeddings.size == 0:
            return []

        scores = cosine_similarity(user_profile.reshape(1, -1), candidate_embeddings).flatten()
        
        top_k_indices = np.argsort(scores)[-k:][::-1]
        return [candidates[i] for i in top_k_indices]
