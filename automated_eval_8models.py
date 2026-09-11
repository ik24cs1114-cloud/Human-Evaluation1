import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path.home() / "myproject"

PRED_DIR = BASE_DIR / "final_test_predictions"
TEST_FILE = Path.home() / "test_queries.json"

OUTPUT_DIR = BASE_DIR / "automated_evaluation_8models"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# MODELS
# ============================================================

MODELS = [
    ("LDA", "LDA_test160.json"),
    ("NMF", "NMF_test160.json"),
    ("KeyBERT", "KeyBERT_test160.json"),
    ("BERTopic", "BERTopic_test160.json"),
    ("FlanT5-v1", "FlanT5-v1_test160.json"),
    ("FlanT5-v2", "FlanT5-v2_test160.json"),
    ("FlanT5-v3", "FlanT5-v3_test160.json"),
    ("Llama3-Groq", "Llama3-Groq_test160.json"),
]


# ============================================================
# LOAD TEST QUERIES
# ============================================================

with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

queries = [x["query"] for x in test_data]

print("=" * 70)
print("AUTOMATED EVALUATION - 8 MODELS")
print("=" * 70)
print("Test queries:", len(queries))


# ============================================================
# LOAD SENTENCE TRANSFORMER
# ============================================================

print("\nLoading embedding model...")

embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# GET HIERARCHY
# ============================================================

def get_hierarchy(item):

    hierarchy = item.get("hierarchy", {})

    if not isinstance(hierarchy, dict):
        return {
            "basic": [],
            "intermediate": [],
            "advanced": []
        }

    return {
        "basic": hierarchy.get("basic", [])
        if isinstance(hierarchy.get("basic", []), list)
        else [],

        "intermediate": hierarchy.get("intermediate", [])
        if isinstance(hierarchy.get("intermediate", []), list)
        else [],

        "advanced": hierarchy.get("advanced", [])
        if isinstance(hierarchy.get("advanced", []), list)
        else []
    }


# ============================================================
# HIERARCHY TO TEXT
# ============================================================

def hierarchy_to_text(hierarchy):

    terms = []

    for level in [
        "basic",
        "intermediate",
        "advanced"
    ]:

        for topic in hierarchy[level]:

            if topic is not None:

                topic = normalize_text(topic)

                if topic:

                    terms.append(topic)

    return " ".join(terms)


# ============================================================
# SEMANTIC SIMILARITY
# ============================================================

def calculate_similarity(query, hierarchy_text):

    if not query or not hierarchy_text:
        return 0.0

    embeddings = embedder.encode(
        [
            query,
            hierarchy_text
        ],
        convert_to_numpy=True
    )

    score = cosine_similarity(
        embeddings[0].reshape(1, -1),
        embeddings[1].reshape(1, -1)
    )[0][0]

    return float(score)


# ============================================================
# TOPIC DRIFT
# ============================================================

def calculate_drift(similarity):

    """
    Topic Drift is defined as:

        Drift = 1 - Semantic Similarity

    Higher drift = more semantic deviation.
    Lower drift = better.
    """

    drift = 1.0 - similarity

    return float(max(0.0, min(1.0, drift)))


# ============================================================
# HIERARCHY QUALITY
# ============================================================

def calculate_hierarchy_quality(hierarchy):

    """
    Reference-free structural quality.

    Components:

    1. Level presence
    2. Non-empty hierarchy
    3. Level progression
    4. Topic diversity

    Score range: 0-1
    Higher is better.
    """

    basic = hierarchy["basic"]
    intermediate = hierarchy["intermediate"]
    advanced = hierarchy["advanced"]

    # --------------------------------------------------------
    # 1. Level presence
    # --------------------------------------------------------

    level_presence = sum(
        [
            bool(basic),
            bool(intermediate),
            bool(advanced)
        ]
    ) / 3.0

    # --------------------------------------------------------
    # 2. Total topics
    # --------------------------------------------------------

    all_topics = (
        basic +
        intermediate +
        advanced
    )

    total_topics = len(all_topics)

    if total_topics == 0:
        return 0.0

    # --------------------------------------------------------
    # 3. Topic diversity
    # --------------------------------------------------------

    normalized_topics = [
        normalize_text(x)
        for x in all_topics
        if normalize_text(x)
    ]

    unique_topics = set(normalized_topics)

    diversity = (
        len(unique_topics) /
        max(1, len(normalized_topics))
    )

    # --------------------------------------------------------
    # 4. Progression
    # --------------------------------------------------------

    progression = 0.0

    if basic and intermediate:
        progression += 0.5

    if intermediate and advanced:
        progression += 0.5

    # --------------------------------------------------------
    # Final hierarchy score
    # --------------------------------------------------------

    score = (
        0.30 * level_presence +
        0.30 * progression +
        0.40 * diversity
    )

    return float(
        max(0.0, min(1.0, score))
    )


# ============================================================
# LOAD MODEL FILE
# ============================================================

def load_predictions(filename):

    path = PRED_DIR / filename

    if not path.exists():

        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return data


# ============================================================
# VERIFY MODEL
# ============================================================

def verify_predictions(model_name, filename, data):

    print()
    print("-" * 70)
    print("VERIFYING:", model_name)
    print("-" * 70)

    print("Records:", len(data))

    if len(data) != 160:
        raise RuntimeError(
            f"{model_name} does not contain 160 records."
        )

    successful = sum(
        1
        for x in data
        if x.get("hierarchy")
    )

    print("Successful:", successful)

    if successful != 160:
        raise RuntimeError(
            f"{model_name} does not have 160 successful predictions."
        )

    # --------------------------------------------------------
    # Match predictions using query text
    # --------------------------------------------------------

    prediction_map = {}

    for item in data:

        query = item.get("query", "").strip()

        if not query:
            continue

        prediction_map[query] = item

    print(
        "Unique prediction queries:",
        len(prediction_map)
    )

    # --------------------------------------------------------
    # Check missing queries
    # --------------------------------------------------------

    missing = []

    for query in queries:

        if query not in prediction_map:
            missing.append(query)

    print(
        "Missing queries:",
        len(missing)
    )

    if missing:

        print("\nFirst missing queries:")

        for q in missing[:5]:

            print(
                repr(q[:150])
            )

        raise RuntimeError(
            f"{model_name} is missing queries."
        )

    # --------------------------------------------------------
    # Check duplicate queries
    # --------------------------------------------------------

    query_counts = {}

    for item in data:

        q = item.get("query", "").strip()

        query_counts[q] = (
            query_counts.get(q, 0) + 1
        )

    duplicates = {
        q: count
        for q, count in query_counts.items()
        if count > 1
    }

    print(
        "Duplicate queries:",
        len(duplicates)
    )

    if duplicates:

        print("\nDuplicate examples:")

        for q, count in list(
            duplicates.items()
        )[:5]:

            print(
                count,
                repr(q[:150])
            )

        raise RuntimeError(
            f"{model_name} contains duplicate queries."
        )

    print(
        "Query alignment: PASS"
    )

    return prediction_map

    # --------------------------------------------------------
    # Query alignment
    # --------------------------------------------------------

    mismatches = 0

    for i in range(160):

        expected = queries[i]

        actual = data[i].get(
            "query",
            ""
        )

        if actual != expected:

            mismatches += 1

            if mismatches <= 5:

                print(
                    "Mismatch:",
                    i + 1
                )

    print(
        "Query mismatches:",
        mismatches
    )

    if mismatches > 0:

        raise RuntimeError(
            f"{model_name} query order does not match test_queries.json."
        )

    print("Verification: PASS")


# ============================================================
# MAIN EVALUATION
# ============================================================

all_results = []


for model_name, filename in MODELS:

    data = load_predictions(filename)

    prediction_map = verify_predictions(
        model_name,
        filename,
        data
    )

    similarities = []
    drifts = []
    hierarchy_scores = []

    print()
    print(
        "Calculating metrics for:",
        model_name
    )

    for i, query in enumerate(queries):

        item = prediction_map[query]

        hierarchy = get_hierarchy(item)

        hierarchy_text = hierarchy_to_text(
            hierarchy
        )

        similarity = calculate_similarity(
            query,
            hierarchy_text
        )

        drift = calculate_drift(
            similarity
        )

        hierarchy_quality = calculate_hierarchy_quality(
            hierarchy
        )

        similarities.append(
            similarity
        )

        drifts.append(
            drift
        )

        hierarchy_scores.append(
            hierarchy_quality
        )

        # -----------------------------------------------
        # Semantic Similarity
        # -----------------------------------------------

        similarity = calculate_similarity(
            query,
            hierarchy_text
        )

        # -----------------------------------------------
        # Topic Drift
        # -----------------------------------------------

        drift = calculate_drift(
            similarity
        )

        # -----------------------------------------------
        # Hierarchy Quality
        # -----------------------------------------------

        hierarchy_quality = calculate_hierarchy_quality(
            hierarchy
        )

        similarities.append(
            similarity
        )

        drifts.append(
            drift
        )

        hierarchy_scores.append(
            hierarchy_quality
        )

    # --------------------------------------------------------
    # Model averages
    # --------------------------------------------------------

    mean_similarity = np.mean(
        similarities
    )

    mean_drift = np.mean(
        drifts
    )

    mean_hierarchy = np.mean(
        hierarchy_scores
    )

    std_similarity = np.std(
        similarities,
        ddof=1
    )

    std_drift = np.std(
        drifts,
        ddof=1
    )

    std_hierarchy = np.std(
        hierarchy_scores,
        ddof=1
    )

    print()
    print(
        f"{model_name}"
    )

    print(
        f"Similarity : "
        f"{mean_similarity:.4f} "
        f"+/- {std_similarity:.4f}"
    )

    print(
        f"Drift      : "
        f"{mean_drift:.4f} "
        f"+/- {std_drift:.4f}"
    )

    print(
        f"Hierarchy  : "
        f"{mean_hierarchy:.4f} "
        f"+/- {std_hierarchy:.4f}"
    )

    all_results.append(
        {
            "Model": model_name,
            "N": len(data),

            "Semantic Similarity":
                mean_similarity,

            "Similarity SD":
                std_similarity,

            "Topic Drift":
                mean_drift,

            "Drift SD":
                std_drift,

            "Hierarchy Quality":
                mean_hierarchy,

            "Hierarchy SD":
                std_hierarchy
        }
    )


# ============================================================
# RESULTS TABLE
# ============================================================

df = pd.DataFrame(
    all_results
)

print()
print("=" * 100)
print("FINAL AUTOMATED RESULTS")
print("=" * 100)

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE CSV
# ============================================================

csv_file = (
    OUTPUT_DIR /
    "automated_results_8models.csv"
)

df.to_csv(
    csv_file,
    index=False
)

print()
print(
    "CSV saved:",
    csv_file
)


# ============================================================
# SAVE PAPER TABLE
# ============================================================

paper_df = df[
    [
        "Model",
        "N",
        "Semantic Similarity",
        "Topic Drift",
        "Hierarchy Quality"
    ]
].copy()

paper_file = (
    OUTPUT_DIR /
    "Table_1_Automated_Evaluation.csv"
)

paper_df.to_csv(
    paper_file,
    index=False
)

print(
    "Paper table saved:",
    paper_file
)


# ============================================================
# FIGURE 1
# SEMANTIC SIMILARITY
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.bar(
    df["Model"],
    df["Semantic Similarity"]
)

plt.xlabel(
    "Models"
)

plt.ylabel(
    "Semantic Similarity"
)

plt.title(
    "Semantic Similarity Across Topic Extraction Models"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

fig1 = (
    OUTPUT_DIR /
    "Figure_1_Semantic_Similarity.png"
)

plt.savefig(
    fig1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 2
# TOPIC DRIFT
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.bar(
    df["Model"],
    df["Topic Drift"]
)

plt.xlabel(
    "Models"
)

plt.ylabel(
    "Topic Drift"
)

plt.title(
    "Topic Drift Across Topic Extraction Models"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

fig2 = (
    OUTPUT_DIR /
    "Figure_2_Topic_Drift.png"
)

plt.savefig(
    fig2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 3
# HIERARCHY QUALITY
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.bar(
    df["Model"],
    df["Hierarchy Quality"]
)

plt.xlabel(
    "Models"
)

plt.ylabel(
    "Hierarchy Quality"
)

plt.title(
    "Hierarchy Quality Across Topic Extraction Models"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

fig3 = (
    OUTPUT_DIR /
    "Figure_3_Hierarchy_Quality.png"
)

plt.savefig(
    fig3,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 100)
print("EVALUATION COMPLETE")
print("=" * 100)

print(
    "Results directory:"
)

print(
    OUTPUT_DIR.resolve()
)

print()
print("Files created:")

print(
    "1.",
    csv_file.name
)

print(
    "2.",
    paper_file.name
)

print(
    "3.",
    fig1.name
)

print(
    "4.",
    fig2.name
)

print(
    "5.",
    fig3.name
)

print("=" * 100)
