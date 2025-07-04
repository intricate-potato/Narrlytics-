
import argparse
from src.data_loader import MINDDataLoader
from src.narrative_extractor import NarrativeExtractor
from src.embedding_generator import NarrativeEmbeddingGenerator
from src.recommender import NarrativeRecommender
from src.evaluation import Evaluator
from src.utils import load_config, set_all_seeds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    set_all_seeds(config['experiment']['random_seed'])

    # Data Loading
    data_loader = MINDDataLoader(args.config, config['data']['mind_dataset_path'], config['data']['mind_version'])
    
    # Narrative Extraction
    narrative_extractor = NarrativeExtractor(config)
    articles_to_process = data_loader.news_data[['NewsID', 'Text']].to_records(index=False)
    actants = narrative_extractor.batch_extract_actants(articles_to_process)

    # Embedding Generation
    embedding_generator = NarrativeEmbeddingGenerator(config)
    actant_embeddings = {aid: embedding_generator.generate_actant_embeddings(a) for aid, a in actants.items()}
    
    training_actant_embeddings = [v for k, v in actant_embeddings.items() if k in data_loader.behaviors_data.loc[data_loader.create_temporal_splits()[0]]['History'].explode().unique()]
    embedding_generator.fit_svd_model(training_actant_embeddings)
    
    narrative_embeddings = embedding_generator.process_article_batch(actants)

    # Recommendation
    recommender = NarrativeRecommender(config, narrative_embeddings)
    
    # Evaluation
    evaluator = Evaluator(config)
    
    # Example Evaluation
    user_id = 'U13740'
    user_history = data_loader.get_user_history(user_id)
    candidates = data_loader.behaviors_data.iloc[0]['Impressions']
    candidate_ids = [c[0] for c in candidates]
    
    recommendations = recommender.recommend_narrative_only(user_id, user_history, candidate_ids, k=10)
    
    print(f"Recommendations for user {user_id}: {recommendations}")

if __name__ == '__main__':
    main()
