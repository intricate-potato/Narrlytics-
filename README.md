# Narrlytics: Narrative-Aware News Recommendation

## 1. Project Overview

This project implements a narrative-aware news recommendation system that leverages Greimas' Actantial Model—a structuralist framework from semiotics—to extract and model the underlying narrative structure of news articles. By representing "who did what to whom" through six functional roles (Subject, Object, Helper, Opponent, Sender, Receiver), the system captures narrative dimensions that traditional content-based and collaborative filtering approaches miss.

## 2. Approach

### Phase 1: Actant Extraction
- Use LLM (LLaMA-8B) to extract six Greimas actants from news titles/abstracts
- Filter noisy extractions to retain semantically meaningful entities
- Output: 20,540 filtered actants from 65,228 news articles

### Phase 2: Heterogeneous Graph Construction
- **Node Types**: Users, News Articles, Actants
- **Edge Types**: 
  - User–News (clicks with temporal decay)
  - News–Actant (containment)
  - Actant–Actant (co-occurrence & co-click)

### Phase 3: HeteroGAT Training
- Heterogeneous Graph Attention Network with type-specific attention
- BPR loss for implicit feedback ranking
- Node features: category embeddings, narrative role encodings, learnable embeddings

## 3. Results

| Model | Prec@5 | nDCG@5 | MRR |
|-------|--------|--------|-----|
| Bipartite GAT | 0.0808 | 0.2342 | 0.2584 |
| **HeteroGAT (Ours)** | **0.0912** | **0.2476** | **0.2724** |

Performance within ~10% of state-of-the-art graph-based news recommenders, with significant room for improvement through richer actant extraction from full article bodies.

## 4. Dataset

Microsoft MIND dataset (MIND-small): 50,000 users, 65,238 news articles, 347,727 click interactions.

**Key Limitation**: MIND provides only titles/abstracts; original URLs have expired, preventing full article body retrieval for richer actant extraction.