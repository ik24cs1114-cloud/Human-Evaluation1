# Ontology-Free LLM Hierarchy Generation

This repository contains the reproducibility materials for **LLM-QTH**, a query-driven framework for generating difficulty-staged educational topic hierarchies using large language models.

LLM-QTH organizes generated topics into three pedagogical levels:

- Basic
- Intermediate
- Advanced

The framework does not require an externally supplied ontology, taxonomy, knowledge graph, prerequisite graph, or manually constructed domain hierarchy at inference time. Here, ontology independence refers specifically to the absence of an external ontology as an input requirement; the underlying language model may still contain implicit domain knowledge acquired during pretraining.

## Cross-Domain Evaluation

The cross-domain evaluation uses queries from **MMLU** and **MedMCQA**.

The automated evaluation contains:

- 100 MMLU queries
- 100 MedMCQA queries

A separate human evaluation was conducted on 50 external queries:

- 25 MMLU queries
- 25 MedMCQA queries
- 3 independent annotators (A1--A3)

The annotators evaluated the generated hierarchies on a 0--3 scale for:

- Topic relevance
- Coherence
- Level appropriateness
- Progression
- Non-redundancy

## Human Evaluation Results

Across the 50 external queries, the mean topic-relevance score was **2.464**.

The overall mean hierarchy-quality scores were:

| Criterion | Mean |
|---|---:|
| Coherence | 2.693 |
| Level appropriateness | 2.613 |
| Progression | 2.647 |
| Non-redundancy | 2.440 |

### Comparison Across External Collections

| Measure | MMLU | MedMCQA | p-value | Cliff's delta |
|---|---:|---:|---:|---:|
| Topic relevance | 2.437 | 2.492 | .248 | -0.192 |
| Coherence | 2.733 | 2.653 | .399 | 0.133 |
| Level appropriateness | 2.533 | 2.693 | .078 | -0.280 |
| Progression | 2.720 | 2.573 | .085 | 0.267 |
| Non-redundancy | 2.413 | 2.467 | .897 | -0.022 |

Two-sided Mann--Whitney U tests conducted at the query level detected no statistically significant between-collection differences for any of the five human-evaluation measures.

These findings provide evidence for cross-domain applicability within the evaluated datasets. The absence of statistically significant differences does not establish statistical equivalence and should not be interpreted as evidence of universal domain independence.

## Repository Contents

The repository contains:

- anonymized human annotation data;
- cross-domain evaluation results;
- statistical analysis scripts;
- dataset- and subject-level summaries;
- inter-annotator agreement results; and
- publication-quality figures.

## Human Annotation Data

The human-evaluation data contain anonymized ratings from three annotators identified as **A1**, **A2**, and **A3**.

Topic-level files contain ratings for topic relevance.

Hierarchy-level files contain ratings for:

- coherence;
- level appropriateness;
- progression; and
- non-redundancy.

All human ratings use a 0--3 scale.

## Inter-Annotator Agreement

Pairwise quadratic-weighted Cohen's kappa was calculated to examine agreement among the three annotators.

The agreement results are reported transparently because item-level agreement was generally low. Therefore, the human-evaluation findings are interpreted primarily as aggregate tendencies rather than as evidence of consistent agreement on individual topic or hierarchy assignments.

## Reproducing the Analysis

Install the required Python packages:

```bash
pip install -r requirements.txt