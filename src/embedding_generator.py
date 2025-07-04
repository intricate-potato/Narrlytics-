import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
import os
import json
import hashlib

class NarrativeEmbeddingGenerator:
    def __init__(self, config):
        self.config = config
        self.embedding_model = SentenceTransformer(config['model']['embedding_model_name'])
        self.svd_models = {actant: TruncatedSVD(n_components=config['model']['reduced_dimension']) for actant in ["Sender", "Receiver", "Subject", "Object", "Helper", "Opponent"]}
        self.cache_dir = os.path.join(config['data']['cache_path'], 'embeddings')
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, article_id):
        return os.path.join(self.cache_dir, f"{hashlib.md5(article_id.encode()).hexdigest()}.json")

    def generate_actant_embeddings(self, actants_dict):
        actant_embeddings = {}
        for actant, names in actants_dict.items():
            if names:
                # Take first actor if multiple exist (following paper)
                actant_embeddings[actant] = self.embedding_model.encode(names[0]).tolist() # Convert to list for JSON
            else:
                actant_embeddings[actant] = np.zeros(self.config['model']['embedding_dimension']).tolist()
        return actant_embeddings

    def fit_svd_model(self, training_actants):
        for actant in self.svd_models.keys():
            # Ensure embeddings are numpy arrays before fitting SVD
            embeddings = [np.array(d[actant]) for d in training_actants if actant in d]
            if embeddings:
                self.svd_models[actant].fit(np.array(embeddings))

    def reduce_dimensions(self, actant_embeddings):
        reduced_embeddings = {}
        for actant, embedding in actant_embeddings.items():
            # Ensure embedding is numpy array before reshaping and transforming
            reduced_embeddings[actant] = self.svd_models[actant].transform(np.array(embedding).reshape(1, -1)).flatten().tolist() # Convert to list for JSON
        return reduced_embeddings

    def create_narrative_embedding(self, reduced_actants):
        # Ensure consistent ordering across all articles
        # Ensure elements are numpy arrays before concatenating
        return np.concatenate([np.array(reduced_actants[actant]) for actant in sorted(reduced_actants.keys())]).tolist() # Convert to list for JSON

    def process_article_batch(self, articles):
        embeddings = {}
        for article_id, actants in tqdm(articles.items(), desc="Generating Embeddings with Caching"):
            cache_path = self._get_cache_path(article_id)
            
            if self.config['data']['cache_embeddings'] and os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    embeddings[article_id] = np.array(json.load(f)) # Convert back to numpy array
                continue

            actant_embeddings = self.generate_actant_embeddings(actants)
            reduced_actants = self.reduce_dimensions(actant_embeddings)
            narrative_embedding = self.create_narrative_embedding(reduced_actants)
            
            embeddings[article_id] = np.array(narrative_embedding) # Store as numpy array in memory
            
            if self.config['data']['cache_embeddings']:
                with open(cache_path, 'w') as f:
                    json.dump(narrative_embedding, f) # Save as list for JSON compatibility
                    
        return embeddings