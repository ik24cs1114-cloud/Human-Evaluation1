# LLM-QTH Reproducibility Package

This repository contains the data splits, generated hierarchies,
evaluation scripts, human-evaluation materials, and supplementary
experiments for the LLM-QTH study.

## Primary benchmark

- Initial query records: 900
- Unique queries after preprocessing/deduplication: 800
- Train / validation / test: 560 / 80 / 160
- Systems evaluated: 8
- Held-out test queries: 160
- Automated system-query evaluations: 1,280
- Human-evaluation queries: 50
- Independent annotators: 5

## Systems

1. LDA
2. NMF
3. KeyBERT
4. BERTopic
5. FlanT5-v1
6. FlanT5-v2
7. FlanT5-v3
8. LLM-QTH (Llama-3.1-8B-Instant via Groq)

## Repository structure

- `data/` - dataset and train/validation/test splits
- `predictions/` - frozen outputs for all eight systems
- `prompts/` - generation prompts
- `generation/` - generation and parsing code
- `automated_evaluation/` - evaluation scripts and results
- `human_evaluation/` - anonymized human evaluation data and analysis
- `verification/` - integrity-check scripts
- `ablation/` - prompt ablation study
- `external_validation/` - MedMCQA/MMLU supplementary validation

## Primary LLM configuration

- Model: Llama-3.1-8B-Instant
- Provider: Groq
- Temperature: 0.3
- Generation: zero-shot
- Output structure: Basic / Intermediate / Advanced

No API keys or personally identifiable annotator information are included.
