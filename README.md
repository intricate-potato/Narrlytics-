# Narrative-Based News Recommendation System

## 1. Project Overview

This project implements a complete Python system for narrative-based news recommendation. It follows the methodology from the paper "Mapping News Narratives Using LLMs and Narrative-Structured Text Embeddings," adapted for use with the Microsoft MIND dataset.

The core idea is to extract the underlying narrative structure (the "who did what to whom") from news articles using a Large Language Model (LLM) and then use these structured narratives to generate more diverse and relevant recommendations.

## 2. System Architecture

The project is organized into the following directory structure:

```
narrative-news-recommendation/
├── configs/              # Experiment configuration (config.yaml)
├── data/
│   ├── mind/             # Raw MIND dataset files will be downloaded here
│   └── cache/            # Cached data to speed up re-runs
│       ├── actants/      # Cached LLM-extracted narratives
│       └── embeddings/   # Cached article embeddings
├── models/               # Saved model files (e.g., SVD models)
├── notebooks/            # Jupyter notebooks for analysis and exploration
├── results/              # Output for metrics, figures, and reports
├── src/                  # All Python source code
├── tests/                # Unit and integration tests
├── main.py               # Main script to run the entire pipeline
├── requirements.txt      # Python dependencies
└── README.md             # This documentation file
```

## 3. Setup and Installation

To set up the environment and run this project, follow these steps:

### Step 1: Install Python Dependencies

All required Python libraries are listed in `requirements.txt`. Install them using pip:

```bash
pip install -r requirements.txt
```

### Step 2: Set Up Ollama

This project uses a local Ollama instance to run the Llama 3 model for narrative extraction. 

1.  **Ensure Ollama is running:** The Ollama application must be running on your machine before executing the Python script.
2.  **Pull the Llama 3 model:** If you haven't already, download the necessary model by running the following command in your terminal:
    ```bash
    ollama pull llama3
    ```

## 4. Running the Experiment

The entire experimental pipeline is orchestrated by the `main.py` script.

### To run the full pipeline:

Execute the following command from the root of the project directory (`/home/intricate-potato/Desktop/llam/news-rec/`):

```bash
python main.py
```

This command will:
1.  Load the configuration from `configs/config.yaml`.
2.  Download the MIND dataset into the `data/mind/` directory if it's not already present.
3.  Extract narrative actants from news articles using Ollama. **This process is cached** in `data/cache/actants/`. If you stop and restart the script, it will resume from where it left off.
4.  Generate narrative-structured embeddings for each article.
5.  (Future implementation) Train recommendation models and run evaluations.

### Configuration

All experiment parameters can be modified in the `configs/config.yaml` file. This includes the dataset version, model names, and hyperparameters.
