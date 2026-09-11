import glob
import numpy as np
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================

OUTPUT_FILE = "human_evaluation_results.xlsx"

# IMPORTANT:
# Replace these with the TRUE mapping used in your experiment.
SYSTEM_TO_MODEL = {
    "System A": "LDA",
    "System B": "FLAN-T5-v2",
    "System C": "FLAN-T5-v3",
    "System D": "FLAN-T5-v1",
    "System E": " KeyBERT",
    "System F": " BERTopic",
    "System G": "NMF",
    "System H": "LLM-QTH",
}

# ============================================================
# NDCG@8
# ============================================================

def dcg_at_k(scores, k=8):
    scores = np.asarray(scores, dtype=float)[:k]

    if len(scores) == 0:
        return np.nan

    gains = (2 ** scores) - 1
    discounts = np.log2(np.arange(2, len(scores) + 2))

    return np.sum(gains / discounts)


def ndcg_at_k(scores, k=8):
    scores = np.asarray(scores, dtype=float)

    if len(scores) == 0:
        return np.nan

    actual_dcg = dcg_at_k(scores, k)

    ideal_scores = np.sort(scores)[::-1]
    ideal_dcg = dcg_at_k(ideal_scores, k)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


# ============================================================
# FIND EXCEL FILES
# ============================================================

files = [
    f for f in glob.glob("*.xlsx")
    if f != OUTPUT_FILE
]

print("Excel files found:", len(files))

if len(files) == 0:
    raise ValueError("No annotator Excel files found.")


# ============================================================
# COLLECT TOPIC RELEVANCE
# ============================================================

topic_records = []

for file in files:

    print("Reading Topic Relevance:", file)

    df = pd.read_excel(
        file,
        sheet_name="Topic Relevance"
    )

    for annotator in range(1, 6):

        rating_col = f"A{annotator}_Relevance"

        if rating_col not in df.columns:
            continue

        temp = df[
            [
                "Query_ID",
                "Query",
                "Anonymous_System",
                "Level",
                "Rank",
                "Topic",
                rating_col
            ]
        ].copy()

        temp = temp.rename(
            columns={
                rating_col: "Relevance"
            }
        )

        temp["Annotator"] = f"A{annotator}"

        # Convert rating to numeric
        temp["Relevance"] = pd.to_numeric(
            temp["Relevance"],
            errors="coerce"
        )

        # Keep only actual ratings
        temp = temp.dropna(
            subset=["Relevance"]
        )

        topic_records.append(temp)


topic_data = pd.concat(
    topic_records,
    ignore_index=True
)

# Remove duplicate ratings in case the same data
# accidentally occur in more than one workbook.
topic_data = topic_data.drop_duplicates(
    subset=[
        "Query_ID",
        "Anonymous_System",
        "Level",
        "Rank",
        "Topic",
        "Annotator"
    ]
)

print("\nTopic ratings collected:", len(topic_data))


# ============================================================
# QUERY-ANNOTATOR MEAN RELEVANCE
# ============================================================

query_relevance = (
    topic_data
    .groupby(
        [
            "Query_ID",
            "Anonymous_System",
            "Annotator"
        ],
        as_index=False
    )
    .agg(
        Mean_Relevance=("Relevance", "mean")
    )
)


# ============================================================
# SYSTEM MEAN RELEVANCE AND SD
# ============================================================

relevance_summary = (
    query_relevance
    .groupby(
        "Anonymous_System"
    )
    .agg(
        **{
            "Mean Relevance":
                ("Mean_Relevance", "mean"),

            "Relevance SD":
                ("Mean_Relevance", "std")
        }
    )
    .reset_index()
)


# ============================================================
# CALCULATE NDCG@8
# ============================================================

ndcg_records = []

for (
    query_id,
    system,
    annotator
), group in topic_data.groupby(
    [
        "Query_ID",
        "Anonymous_System",
        "Annotator"
    ]
):

    # Preserve hierarchy level first,
    # then rank within level.
    level_order = {
        "basic": 0,
        "intermediate": 1,
        "advanced": 2
    }

    group = group.copy()

    group["Level_Order"] = (
        group["Level"]
        .astype(str)
        .str.lower()
        .map(level_order)
        .fillna(99)
    )

    group["Rank"] = pd.to_numeric(
        group["Rank"],
        errors="coerce"
    )

    group = group.sort_values(
        ["Level_Order", "Rank"]
    )

    relevance_scores = (
        group["Relevance"]
        .dropna()
        .tolist()
    )

    score = ndcg_at_k(
        relevance_scores,
        k=8
    )

    ndcg_records.append(
        {
            "Query_ID": query_id,
            "Anonymous_System": system,
            "Annotator": annotator,
            "NDCG@8": score
        }
    )


ndcg_data = pd.DataFrame(
    ndcg_records
)


ndcg_summary = (
    ndcg_data
    .groupby(
        "Anonymous_System"
    )
    .agg(
        **{
            "NDCG@8":
                ("NDCG@8", "mean"),

            "NDCG SD":
                ("NDCG@8", "std")
        }
    )
    .reset_index()
)


# ============================================================
# COLLECT HIERARCHY RATINGS
# ============================================================

hierarchy_records = []

metrics = [
    "Coherence",
    "Level_Appropriateness",
    "Progression",
    "Non_Redundancy"
]

for file in files:

    print("Reading Hierarchy Evaluation:", file)

    df = pd.read_excel(
        file,
        sheet_name="Hierarchy Evaluation"
    )

    for annotator in range(1, 6):

        required_columns = [
            f"A{annotator}_{metric}"
            for metric in metrics
        ]

        if not all(
            col in df.columns
            for col in required_columns
        ):
            continue

        temp = df[
            [
                "Query_ID",
                "Query",
                "Anonymous_System"
            ] + required_columns
        ].copy()

        rename_dict = {
            f"A{annotator}_{metric}": metric
            for metric in metrics
        }

        temp = temp.rename(
            columns=rename_dict
        )

        temp["Annotator"] = f"A{annotator}"

        # Convert ratings to numeric
        for metric in metrics:

            temp[metric] = pd.to_numeric(
                temp[metric],
                errors="coerce"
            )

        # Keep rows containing at least one rating
        temp = temp.dropna(
            subset=metrics,
            how="all"
        )

        hierarchy_records.append(temp)


hierarchy_data = pd.concat(
    hierarchy_records,
    ignore_index=True
)


# Remove duplicate ratings
hierarchy_data = hierarchy_data.drop_duplicates(
    subset=[
        "Query_ID",
        "Anonymous_System",
        "Annotator"
    ]
)

print(
    "Hierarchy ratings collected:",
    len(hierarchy_data)
)


# ============================================================
# HIERARCHY OVERALL
# ============================================================

# Composite score:
# mean of the four hierarchy dimensions.

hierarchy_data["Hierarchy Overall"] = (
    hierarchy_data[
        [
            "Coherence",
            "Level_Appropriateness",
            "Progression",
            "Non_Redundancy"
        ]
    ]
    .mean(
        axis=1,
        skipna=False
    )
)


# ============================================================
# SYSTEM-LEVEL HIERARCHY RESULTS
# ============================================================

hierarchy_summary = (
    hierarchy_data
    .groupby(
        "Anonymous_System"
    )
    .agg(
        Coherence=(
            "Coherence",
            "mean"
        ),

        **{
            "Level Appropriateness": (
                "Level_Appropriateness",
                "mean"
            ),

            "Progression": (
                "Progression",
                "mean"
            ),

            "Non-Redundancy": (
                "Non_Redundancy",
                "mean"
            ),

            "Hierarchy Overall": (
                "Hierarchy Overall",
                "mean"
            )
        }
    )
    .reset_index()
)


# ============================================================
# MERGE RESULTS
# ============================================================

final = relevance_summary.merge(
    ndcg_summary,
    on="Anonymous_System",
    how="outer"
)

final = final.merge(
    hierarchy_summary,
    on="Anonymous_System",
    how="outer"
)


# ============================================================
# ADD MODEL NAME
# ============================================================

final.insert(
    1,
    "Model",
    final["Anonymous_System"].map(
        SYSTEM_TO_MODEL
    )
)


# ============================================================
# ROUND RESULTS
# ============================================================

numeric_columns = [
    "Mean Relevance",
    "Relevance SD",
    "NDCG@8",
    "NDCG SD",
    "Coherence",
    "Level Appropriateness",
    "Progression",
    "Non-Redundancy",
    "Hierarchy Overall"
]

final[numeric_columns] = (
    final[numeric_columns]
    .round(4)
)


# ============================================================
# DISPLAY FINAL TABLE
# ============================================================

print("\n")
print("=" * 120)
print("FINAL HUMAN EVALUATION TABLE")
print("=" * 120)

print(
    final.to_string(
        index=False
    )
)


# ============================================================
# SAVE EVERYTHING TO EXCEL
# ============================================================

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    final.to_excel(
        writer,
        sheet_name="Final Results",
        index=False
    )

    query_relevance.to_excel(
        writer,
        sheet_name="Query Relevance",
        index=False
    )

    ndcg_data.to_excel(
        writer,
        sheet_name="NDCG Detail",
        index=False
    )

    hierarchy_data.to_excel(
        writer,
        sheet_name="Hierarchy Detail",
        index=False
    )


print(
    "\nSaved successfully:",
    OUTPUT_FILE
)