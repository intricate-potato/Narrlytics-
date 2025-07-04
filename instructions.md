# Complete Instructions for Narrative-Based News Recommendation System Using MIND Dataset

## Project Overview
Implement a complete Python system that extracts narrative structures from news articles using the Actantial Model and leverages these structures for improved news recommendations. This project follows the methodology from "Mapping News Narratives Using LLMs and Narrative-Structured Text Embeddings" adapted for the recommendation task on Microsoft's MIND dataset.

## System Architecture

### Directory Structure
```
narrative-news-recommendation/
├── data/
│   ├── mind/
│   │   ├── train/
│   │   ├── dev/
│   │   └── test/
│   └── cache/
│       ├── actants/
│       └── embeddings/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── narrative_extractor.py
│   ├── embedding_generator.py
│   ├── recommender.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── utils.py
├── models/
│   ├── svd_models/
│   └── checkpoints/
├── results/
│   ├── figures/
│   ├── metrics/
│   └── reports/
├── configs/
│   └── config.yaml
├── tests/
│   └── test_*.py
├── notebooks/
│   └── analysis.ipynb
├── requirements.txt
├── setup.py
├── main.py
└── README.md
```

## Detailed Implementation Instructions

### 1. Configuration File (configs/config.yaml)
```yaml
# Complete configuration for narrative news recommendation
data:
  mind_dataset_path: "data/mind/"
  mind_version: "small"  # "small" or "large"
  mind_url: "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip"
  max_title_length: 30
  max_abstract_length: 200
  combine_title_abstract: true
  cache_extracted_actants: true
  cache_embeddings: true

model:
  # LLM Configuration
  llm_model_name: "meta-llama/Meta-Llama-3-8B-Instruct"
  llm_device: "cuda"  # or "cpu"
  llm_max_length: 512
  llm_temperature: 0.0  # Deterministic output
  llm_do_sample: false
  llm_max_new_tokens: 256
  
  # Embedding Configuration
  embedding_model_name: "intfloat/e5-large"
  embedding_dimension: 1024
  reduced_dimension: 34
  
  # SVD Configuration
  svd_random_state: 42
  svd_n_iter: 5

recommendation:
  methods: ["narrative", "content", "hybrid", "diverse"]
  top_k: [5, 10, 20]
  hybrid_alpha: 0.5
  diversity_lambda: 0.3
  batch_size: 128

training:
  num_epochs: 10
  learning_rate: 0.001
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  negative_sampling_ratio: 4

evaluation:
  metrics: ["auc", "mrr", "ndcg@5", "ndcg@10", "precision@5", "precision@10"]
  diversity_metrics: ["narrative_diversity", "intra_list_distance", "coverage"]
  
visualization:
  umap_n_neighbors: 15
  umap_min_dist: 0.1
  umap_n_components: 2
  umap_metric: "cosine"
  n_clusters: 18
  clustering_method: "agglomerative"

experiment:
  random_seed: 42
  num_workers: 4
  use_gpu: true
  checkpoint_interval: 1000
  log_interval: 100
```

### 2. Requirements File (requirements.txt)
```
# Core dependencies
torch>=2.0.0
transformers>=4.35.0
sentence-transformers>=2.2.2
accelerate>=0.24.0

# Data processing
numpy>=1.24.3
pandas>=2.0.3
scikit-learn>=1.3.0
scipy>=1.11.0

# Visualization
matplotlib>=3.7.2
seaborn>=0.12.2
plotly>=5.17.0
umap-learn>=0.5.4

# Utilities
tqdm>=4.66.0
pyyaml>=6.0.1
requests>=2.31.0
joblib>=1.3.0

# Development
jupyter>=1.0.0
pytest>=7.4.0
black>=23.7.0
pylint>=2.17.0
```

### 3. Data Loader Implementation (src/data_loader.py)
```python
"""
Implement a comprehensive data loader for MIND dataset with the following specifications:

Class: MINDDataLoader

Initialization parameters:
- config_path: Path to configuration file
- data_dir: Base directory for data storage
- version: 'small' or 'large' MIND dataset
- download: Whether to download data if not present

Main Methods:

1. download_mind_dataset():
   - Check if data exists in data_dir
   - Download from Microsoft servers if missing
   - Extract zip files
   - Verify file integrity
   - Handle download failures gracefully

2. load_news_data():
   - Read news.tsv with proper column names
   - Columns: [NewsID, Category, SubCategory, Title, Abstract, URL, Title_Entities, Abstract_Entities]
   - Handle missing abstracts (some articles only have titles)
   - Create combined text field (title + abstract)
   - Return pandas DataFrame with processed data

3. load_behaviors_data():
   - Read behaviors.tsv with columns: [ImpressionID, UserID, Time, History, Impressions]
   - Parse history field: "N12345 N23456" -> ["N12345", "N23456"]
   - Parse impressions: "N12345-1 N23456-0" -> [("N12345", 1), ("N23456", 0)]
   - Convert timestamps to datetime objects
   - Return processed DataFrame

4. create_user_item_interactions():
   - Extract positive (clicked) and negative (shown but not clicked) samples
   - Create user-item interaction matrix
   - Handle implicit feedback
   - Return sparse matrix format for efficiency

5. get_article_text(news_id):
   - Retrieve full text for a given news ID
   - Combine title and abstract based on config
   - Handle missing articles gracefully
   - Apply text preprocessing if needed

6. create_temporal_splits():
   - Split data based on timestamps (not random)
   - Ensure temporal validity (test after validation after train)
   - Maintain user consistency across splits
   - Return train/val/test indices

7. get_user_history(user_id, max_history=50):
   - Retrieve clicked articles for user
   - Sort by timestamp
   - Limit to most recent max_history items
   - Return list of news IDs

8. create_candidate_sets():
   - For each impression, extract shown articles
   - Separate positive and negative samples
   - Create evaluation sets
   - Return dictionary of candidates per impression

Additional utility methods:
- save_processed_data(): Cache processed data
- load_processed_data(): Load from cache
- get_statistics(): Dataset statistics (users, items, interactions)
- filter_cold_users(min_interactions=5): Remove users with few interactions
- filter_cold_items(min_interactions=5): Remove items with few interactions
"""
```

### 4. Narrative Extractor Implementation (src/narrative_extractor.py)
```python
"""
Implement the Actantial Model extraction using LLMs.

Class: NarrativeExtractor

Initialization:
- Load Llama-3-8B-Instruct model using transformers
- Configure for deterministic generation (temperature=0)
- Set up GPU/CPU based on availability
- Initialize cache directory for storing extractions

Core Components:

1. ACTANTIAL_PROMPT (class constant):
   Exact prompt from the paper:
   ```
   According to the Actantial Model by Greimas with the actant label set 
   ["Sender", "Receiver", "Subject", "Object", "Helper", "Opponent"], 
   the actants are defined as follows:
   * Subject: The character who carries out the action and desires the Object.
   * Object: The character or thing that is desired.
   * Sender: The character who initiates the action and communicates the Object.
   * Receiver: The character who receives the action or the Object.
   * Helper: The character who assists the Subject in achieving its goal.
   * Opponent: The character who opposes the Subject in achieving its goal.

   Based on this Actantial Model and the actant label set, please recognize 
   the actants in the given article.

   Article: {article_text}

   Question: What are the main actants in the text? Provide the answer in the 
   following JSON format: {"Actant Label": ["Actant Name"]}. If there is no 
   corresponding actant, return the following empty list: {"Actant Label": []}.

   Answer:
   ```

2. extract_actants(article_text: str) -> Dict[str, List[str]]:
   - Format prompt with article text
   - Generate response using LLM
   - Parse JSON from response
   - Validate all 6 actants are present
   - Handle parsing errors
   - Return standardized actant dictionary

3. batch_extract_actants(articles: List[Tuple[str, str]]) -> Dict[str, Dict]:
   - Process multiple articles efficiently
   - Implement batching for GPU efficiency
   - Show progress bar with tqdm
   - Cache results after each batch
   - Handle failures without stopping
   - Return {article_id: actants_dict}

4. validate_actants(actants: Dict) -> bool:
   - Check all 6 actant types present
   - Verify JSON structure
   - Ensure actant values are lists
   - Log validation errors
   - Return validation status

5. post_process_actants(actants: Dict) -> Dict:
   - Standardize actant names (handle variations)
   - Remove duplicates within each actant type
   - Handle common extraction errors
   - Fill missing actants with empty lists
   - Return cleaned actants

6. cache_actants(article_id: str, actants: Dict):
   - Create hash-based filename
   - Save to cache directory
   - Use JSON format for readability
   - Implement atomic writes
   - Handle concurrent access

7. load_cached_actants(article_id: str) -> Optional[Dict]:
   - Check cache existence
   - Load and validate cached data
   - Return None if not found
   - Handle corrupted cache files

Error Handling:
- Implement exponential backoff for LLM failures
- Log all errors with context
- Provide fallback for failed extractions
- Monitor extraction success rate

Performance Optimizations:
- Use batch generation where possible
- Implement multiprocessing for CPU-bound parsing
- Optimize prompt length
- Reuse model attention cache
"""
```

### 5. Embedding Generator Implementation (src/embedding_generator.py)
```python
"""
Generate narrative-structured embeddings following the paper's methodology.

Class: NarrativeEmbeddingGenerator

Initialization:
- Load E5-large model via sentence-transformers
- Initialize TruncatedSVD for dimension reduction
- Set up caching system
- Configure device (GPU/CPU)

Core Methods:

1. generate_actant_embeddings(actants_dict: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
   - For each actant type (Subject, Object, etc.):
     * Take first actor if multiple exist (following paper)
     * Generate 1024-dimensional embedding using E5-large
     * Handle empty actants with zero vectors
   - Return dictionary of actant embeddings

2. reduce_dimensions(actant_embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
   - Apply SVD to reduce from 1024 to 34 dimensions
   - Fit SVD on training data only
   - Transform validation/test data using fitted SVD
   - Save SVD model for consistency
   - Return reduced embeddings

3. create_narrative_embedding(reduced_actants: Dict[str, np.ndarray]) -> np.ndarray:
   - Concatenate in fixed order: [Subject, Object, Sender, Receiver, Helper, Opponent]
   - Ensure consistent ordering across all articles
   - Result: 6 * 34 = 204 dimensional vector
   - Normalize if specified in config
   - Return final narrative embedding

4. process_article_batch(articles: List[Tuple[str, Dict]], fitted_svd=None) -> Dict[str, np.ndarray]:
   - Process multiple articles efficiently
   - Batch embedding generation
   - Apply dimension reduction
   - Cache results
   - Return {article_id: narrative_embedding}

5. fit_svd_model(training_actants: List[Dict[str, np.ndarray]]):
   - Collect all training actant embeddings
   - Fit separate SVD for each actant type
   - Save fitted models
   - Log explained variance ratios
   - Return fitted SVD models

6. save_embeddings(embeddings: Dict[str, np.ndarray], filepath: str):
   - Save as compressed numpy archive
   - Include metadata (dimensions, actant order)
   - Implement versioning
   - Verify save integrity

7. load_embeddings(filepath: str) -> Dict[str, np.ndarray]:
   - Load embeddings with validation
   - Check dimension consistency
   - Handle version mismatches
   - Return embeddings dictionary

Additional Utilities:
- compute_embedding_statistics(): Mean, std, sparsity
- visualize_embedding_distribution(): For debugging
- validate_embedding_quality(): Sanity checks
- benchmark_embedding_speed(): Performance metrics
"""
```

### 6. Recommender Implementation (src/recommender.py)
```python
"""
Implement narrative-aware recommendation system.

Class: NarrativeRecommender

Main Components:

1. User Profile Generation:
   build_user_narrative_profile(user_history: List[str], article_embeddings: Dict) -> np.ndarray:
   - Retrieve narrative embeddings for user's clicked articles
   - Weight by recency (optional)
   - Average embeddings (or other aggregation)
   - Handle cold-start users
   - Normalize profile vector

2. Scoring Functions:
   narrative_similarity(user_profile: np.ndarray, article_embedding: np.ndarray) -> float:
   - Compute cosine similarity
   - Apply any transformations
   - Return normalized score

   content_similarity(user_profile: np.ndarray, article_content: np.ndarray) -> float:
   - Traditional content-based similarity
   - Use title/abstract embeddings
   - Return normalized score

   hybrid_score(narrative_sim: float, content_sim: float, alpha: float) -> float:
   - Weighted combination
   - Alpha controls narrative vs content weight
   - Return final score

3. Recommendation Methods:
   recommend_narrative_only(user_id: str, candidates: List[str], k: int) -> List[str]:
   - Build user narrative profile
   - Score all candidates by narrative similarity
   - Return top-k articles

   recommend_content_only(user_id: str, candidates: List[str], k: int) -> List[str]:
   - Traditional content-based approach
   - Use BERT embeddings of title/abstract
   - Return top-k articles

   recommend_hybrid(user_id: str, candidates: List[str], k: int, alpha: float) -> List[str]:
   - Combine narrative and content scores
   - Apply hybrid scoring
   - Return top-k articles

   recommend_diverse(user_id: str, candidates: List[str], k: int, lambda_div: float) -> List[str]:
   - Maximize relevance and diversity
   - Use MMR (Maximal Marginal Relevance)
   - Balance narrative similarity and diversity
   - Return diverse top-k articles

4. Diversity Optimization:
   compute_narrative_diversity(articles: List[str], embeddings: Dict) -> float:
   - Calculate pairwise distances
   - Average distance as diversity measure
   - Return diversity score

   mmr_rerank(scores: Dict[str, float], embeddings: Dict, lambda_param: float) -> List[str]:
   - Implement Maximal Marginal Relevance
   - Iteratively select diverse items
   - Balance relevance and diversity

5. Batch Processing:
   recommend_batch(user_ids: List[str], candidate_sets: Dict, method: str) -> Dict:
   - Process multiple users efficiently
   - Parallelize where possible
   - Return recommendations for all users

6. Online Learning (Optional):
   update_user_profile(user_id: str, clicked_article: str):
   - Incrementally update user profile
   - Implement forgetting factor
   - Save updated profile

Performance Optimizations:
- Pre-compute candidate embeddings
- Use approximate nearest neighbor search for large candidate sets
- Implement caching for user profiles
- Batch similarity computations
"""
```

### 7. Evaluation Implementation (src/evaluation.py)
```python
"""
Comprehensive evaluation framework for narrative recommendations.

Class: Evaluator

Standard Recommendation Metrics:

1. calculate_auc(predictions: Dict, labels: Dict) -> float:
   - Area Under ROC Curve
   - Handle class imbalance
   - Return AUC score

2. calculate_mrr(recommendations: Dict, ground_truth: Dict) -> float:
   - Mean Reciprocal Rank
   - Find first relevant item position
   - Average across users

3. calculate_ndcg(recommendations: Dict, ground_truth: Dict, k: int) -> float:
   - Normalized Discounted Cumulative Gain
   - Handle graded relevance
   - Apply position discount

4. calculate_precision_recall(recommendations: Dict, ground_truth: Dict, k: int) -> Tuple[float, float]:
   - Precision@k and Recall@k
   - Handle varying ground truth sizes

Narrative-Specific Metrics:

1. calculate_narrative_diversity(recommendations: Dict, embeddings: Dict) -> float:
   - Average pairwise narrative distance
   - Within recommendation lists
   - Return diversity score

2. calculate_narrative_coverage(recommendations: Dict, actants: Dict) -> float:
   - How many different narrative patterns covered
   - Cluster narratives first
   - Measure cluster coverage

3. calculate_viewpoint_diversity(recommendations: Dict, actants: Dict) -> float:
   - Diversity of Subject/Object combinations
   - Measure perspective variety
   - Return diversity metric

4. calculate_intra_list_distance(recommendations: List[str], embeddings: Dict) -> float:
   - Average distance between recommended items
   - Use narrative embeddings
   - Higher = more diverse

Comparative Analysis:

1. compare_methods(results: Dict[str, Dict]) -> pd.DataFrame:
   - Statistical comparison of methods
   - Include significance tests
   - Generate comparison table

2. perform_ablation_study(base_results: Dict, ablation_results: Dict) -> Dict:
   - Compare full model vs ablations
   - Identify important components
   - Return impact analysis

3. analyze_user_segments(results: Dict, user_features: Dict) -> Dict:
   - Performance by user type
   - Cold vs warm users
   - Activity level segments

Temporal Analysis:

1. evaluate_temporal_performance(results: Dict, timestamps: Dict) -> Dict:
   - Performance over time
   - Concept drift detection
   - Return temporal metrics

2. analyze_narrative_evolution(narratives: Dict, timestamps: Dict) -> Dict:
   - How narratives change over time
   - Emerging vs declining patterns
   - Return evolution analysis

Report Generation:

1. generate_evaluation_report(results: Dict, output_path: str):
   - Comprehensive markdown report
   - Include all metrics
   - Add visualizations
   - Statistical significance

2. create_latex_tables(results: Dict, output_path: str):
   - Publication-ready tables
   - Proper formatting
   - Include significance markers

Error Analysis:

1. analyze_failure_cases(predictions: Dict, ground_truth: Dict) -> Dict:
   - Identify systematic failures
   - Categorize error types
   - Return error analysis

2. analyze_cold_start_performance(results: Dict, user_history_length: Dict) -> Dict:
   - Performance vs history length
   - Cold start identification
   - Return cold start analysis
"""
```

### 8. Visualization Implementation (src/visualization.py)
```python
"""
Create comprehensive visualizations for narrative analysis.

Class: NarrativeVisualizer

UMAP Visualization:

1. create_narrative_space_visualization(embeddings: Dict, metadata: Dict):
   - Apply UMAP to narrative embeddings
   - Color by category/source/cluster
   - Interactive plotly visualization
   - Save static and interactive versions

2. plot_cluster_analysis(embeddings: Dict, n_clusters: int):
   - Apply agglomerative clustering
   - Visualize dendrogram
   - Show cluster assignments
   - Plot cluster characteristics

Actant Analysis:

1. plot_actant_distribution(actants: Dict, groupby: str):
   - Bar charts of common actants
   - Group by category/source
   - Show relative frequencies
   - Create grid of subplots

2. create_actant_network(actants: Dict, min_connections: int):
   - Network graph of actant relationships
   - Nodes = actors, edges = co-occurrence
   - Use networkx for layout
   - Interactive visualization

3. plot_actant_heatmap(actants: Dict, sources: List[str]):
   - Compare actant usage across sources
   - Heatmap visualization
   - Statistical significance markers

Performance Visualization:

1. plot_method_comparison(results: Dict):
   - Bar charts for each metric
   - Error bars for confidence intervals
   - Statistical significance markers
   - Publication-ready formatting

2. create_learning_curves(training_history: Dict):
   - Plot metrics over epochs
   - Training vs validation
   - Identify overfitting
   - Multiple metrics subplot

3. plot_ablation_results(ablation_results: Dict):
   - Impact of each component
   - Tornado/waterfall chart
   - Relative importance

Diversity Analysis:

1. plot_diversity_comparison(diversity_metrics: Dict):
   - Compare diversity across methods
   - Box plots or violin plots
   - Statistical comparisons

2. visualize_recommendation_diversity(user_recs: Dict, embeddings: Dict):
   - t-SNE/UMAP of recommendations
   - Show diversity visually
   - User-specific plots

Temporal Visualization:

1. plot_narrative_evolution(narratives_over_time: Dict):
   - Stream graph or alluvial diagram
   - Show narrative flow over time
   - Interactive time slider

2. plot_temporal_performance(metrics_over_time: Dict):
   - Line plots with confidence bands
   - Multiple metrics
   - Highlight significant changes

User Analysis:

1. plot_user_preference_clusters(user_profiles: Dict):
   - Cluster users by narrative preference
   - Visualize cluster characteristics
   - Show user distribution

2. create_user_journey_visualization(user_history: List, embeddings: Dict):
   - Show evolution of user preferences
   - Narrative trajectory over time
   - Interactive path visualization

Report Generation:

1. create_visualization_report(output_dir: str):
   - Generate all visualizations
   - Create HTML report
   - Include interactive elements
   - Export as PDF option

Utility Functions:
- setup_plotting_style(): Consistent aesthetics
- save_figure(): Multiple format export
- create_subplot_grid(): Flexible layouts
- add_significance_markers(): Statistical annotations
"""
```

### 9. Main Experimental Pipeline (main.py)
```python
"""
Main script orchestrating all experiments.

Structure:

1. Argument Parsing:
   - --config: Path to configuration file
   - --experiment: Type of experiment to run
   - --gpu: GPU device ID
   - --resume: Resume from checkpoint
   - --debug: Debug mode with reduced data

2. Initialization:
   - Load configuration
   - Set random seeds
   - Initialize logging
   - Create output directories
   - Load checkpoint if resuming

3. Data Pipeline:
   - Initialize MINDDataLoader
   - Download/load dataset
   - Create temporal splits
   - Generate candidate sets
   - Log dataset statistics

4. Narrative Extraction Pipeline:
   - Initialize NarrativeExtractor
   - Check cache for existing extractions
   - Extract actants for all articles
   - Validate extraction quality
   - Save extracted actants

5. Embedding Generation Pipeline:
   - Initialize NarrativeEmbeddingGenerator
   - Fit SVD on training data
   - Generate embeddings for all articles
   - Validate embedding quality
   - Cache embeddings

6. Experiments:

   A. Baseline Experiments:
      - Random recommendations
      - Popularity-based
      - Content-based (BERT)
      - Log baseline results

   B. Narrative Experiments:
      - Narrative-only recommendations
      - Hybrid recommendations (vary alpha)
      - Diversity-optimized recommendations
      - Log narrative results

   C. Ablation Studies:
      - Remove each actant type
      - Vary embedding dimensions
      - Different embedding models
      - Different aggregation methods

   D. Analysis Experiments:
      - User preference clustering
      - Narrative pattern analysis
      - Temporal evolution study
      - Source comparison

7. Evaluation:
   - Run all evaluation metrics
   - Generate comparison tables
   - Perform statistical tests
   - Create evaluation report

8. Visualization:
   - Generate all plots
   - Create interactive visualizations
   - Export publication-ready figures
   - Generate HTML report

9. Results Compilation:
   - Aggregate all results
   - Create final report
   - Save all artifacts
   - Create reproducibility package

Main Functions:

run_baseline_experiments(data_loader, evaluator):
   - Implement all baseline methods
   - Evaluate on test set
   - Return results dictionary

run_narrative_experiments(data_loader, embeddings, evaluator):
   - Test all narrative methods
   - Vary hyperparameters
   - Return results dictionary

run_ablation_studies(data_loader, embeddings, evaluator):
   - Test component importance
   - Systematic removal/modification
   - Return ablation results

run_analysis_experiments(data_loader, embeddings, actants):
   - User clustering
   - Pattern mining
   - Temporal analysis
   - Return analysis results

generate_final_report(all_results, output_dir):
   - Compile all findings
   - Create structured report
   - Include best configurations
   - Add reproducibility info

Error Handling:
- Checkpoint after each major step
- Graceful failure recovery
- Comprehensive error logging
- Email notification on completion/failure

Performance Monitoring:
- Track runtime for each component
- Monitor memory usage
- Log GPU utilization
- Create performance report
"""
```

### 10. Utility Functions (src/utils.py)
```python
"""
Utility functions used across the project.

Caching Utilities:
- create_cache_key(text: str) -> str: Generate consistent cache keys
- save_to_cache(data: Any, filepath: str): Atomic cache writing
- load_from_cache(filepath: str) -> Any: Safe cache loading
- clear_cache(cache_dir: str): Clean up old cache files

Text Processing:
- preprocess_text(text: str) -> str: Clean and normalize text
- truncate_text(text: str, max_length: int) -> str: Smart truncation
- combine_title_abstract(title: str, abstract: str) -> str: Merge text fields

Evaluation Utilities:
- calculate_confidence_interval(scores: List[float]) -> Tuple[float, float]
- perform_significance_test(scores1: List[float], scores2: List[float]) -> float
- create_confusion_matrix(predictions: Dict, labels: Dict) -> np.ndarray

Logging Configuration:
- setup_logging(log_dir: str, level: str): Configure comprehensive logging
- get_logger(name: str) -> logging.Logger: Get configured logger
- log_experiment_config(config: Dict): Log all hyperparameters

GPU/Memory Management:
- get_gpu_memory_usage() -> Dict: Monitor GPU memory
- clear_gpu_cache(): Free GPU memory
- estimate_memory_requirement(n_articles: int) -> int

Reproducibility:
- set_all_seeds(seed: int): Set seeds for all libraries
- save_experiment_info(info: Dict, output_dir: str): Save for reproduction
- create_requirements_snapshot(): Capture exact package versions

Data Utilities:
- create_user_item_matrix(interactions: List) -> sp.sparse.csr_matrix
- negative_sampling(positive_samples: List, n_items: int, ratio: int) -> List
- temporal_train_test_split(data: pd.DataFrame, test_days: int) -> Tuple

Parallel Processing:
- parallel_apply(func: Callable, items: List, n_workers: int) -> List
- batch_iterator(items: List, batch_size: int) -> Iterator
- multiprocess_safe_cache(func: Callable) -> Callable: Decorator for safe caching
"""
```

### 11. Testing Suite (tests/)
Create comprehensive tests for each module:
- test_data_loader.py: Test data loading and preprocessing
- test_narrative_extractor.py: Test actant extraction with examples
- test_embedding_generator.py: Test embedding dimensions and quality
- test_recommender.py: Test recommendation logic
- test_evaluation.py: Verify metric calculations
- test_integration.py: End-to-end pipeline tests

### 12. Analysis Notebook (notebooks/analysis.ipynb)
Create Jupyter notebook with:
- Data exploration and statistics
- Actant extraction examples and validation
- Embedding space visualization
- Recommendation case studies
- Error analysis
- Interactive visualizations

## Expected Outputs

### Performance Metrics (results/metrics/)
- evaluation_results.csv: All metrics for all methods
- statistical_tests.csv: Significance test results  
- ablation_results.csv: Component importance
- best_configurations.json: Optimal hyperparameters

### Visualizations (results/figures/)
- narrative_space_umap.png: 2D projection of narratives
- method_comparison.png: Performance bar charts
- diversity_analysis.png: Diversity metrics visualization
- cluster_analysis.png: Narrative clusters
- temporal_evolution.png: Narratives over time
- user_preferences.png: User clustering

### Reports (results/reports/)
- final_report.md: Comprehensive findings
- technical_report.pdf: Detailed methodology
- executive_summary.md: High-level insights

### Models (models/)
- svd_models/*.pkl: Fitted SVD transformers
- cached_actants/: Extracted actants
- cached_embeddings/: Generated embeddings
- user_profiles/: User narrative preferences

## Performance Requirements

- Actant extraction: ~1-2 seconds per article (with GPU)
- Embedding generation: <100ms per article  
- Recommendation generation: <50ms per user
- Full pipeline on MIND-small: <3 hours
- Memory usage: <16GB RAM, <8GB GPU memory

## Key Implementation Notes

1. **Caching Strategy**: Aggressively cache LLM outputs and embeddings
2. **Batch Processing**: Process in batches for GPU efficiency
3. **Error Recovery**: Checkpoint frequently, handle failures gracefully
4. **Reproducibility**: Set all random seeds, log everything
5. **Scalability**: Design for MIND-large from the start

This implementation creates a complete research system for narrative-based news recommendation, enabling thorough comparison with baselines and comprehensive analysis of the narrative approach's benefits.
