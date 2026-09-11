from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import wilcoxon, spearmanr
from statsmodels.stats.inter_rater import fleiss_kappa
from statsmodels.stats.multitest import multipletests
import krippendorff


ANNOTATORS = ["A1", "A2", "A3", "A4", "A5"]

CRITERIA = [
    "Coherence",
    "Level_Appropriateness",
    "Progression",
    "Non_Redundancy",
]

MODEL_MAP = {
    "System A": "LDA",
    "System B": "FLAN-T5-v2",
    "System C": "FLAN-T5-v3",
    "System D": "FLAN-T5-v1",
    "System E": "KeyBERT",
    "System F": "BERTopic",
    "System G": "NMF",
    "System H": "LLM-QTH",
}

LEVEL_ORDER = {
    "basic": 0,
    "intermediate": 1,
    "advanced": 2,
}


def find_files(folder):
    files = [
        p for p in folder.glob("*.xlsx")
        if "annotator" in p.name.lower()
        and not p.name.startswith("~$")
        and "human_evaluation_results" not in p.name.lower()
    ]

    if len(files) != 5:
        raise RuntimeError(
            f"Expected 5 annotator files, but found {len(files)}:\n"
            + "\n".join(p.name for p in files)
        )

    return sorted(files)


def detect_annotator(topic):
    counts = {}

    for a in ANNOTATORS:
        col = f"{a}_Relevance"

        counts[a] = pd.to_numeric(
            topic[col],
            errors="coerce"
        ).notna().sum()

    present = [a for a, n in counts.items() if n > 0]

    if len(present) != 1:
        raise RuntimeError(
            f"Could not identify annotator uniquely: {counts}"
        )

    return present[0]


def load_data(folder):

    topic_parts = {}
    hierarchy_parts = {}

    files = find_files(folder)

    print("\nAnnotator files found:", len(files))

    for file in files:

        topic = pd.read_excel(
            file,
            sheet_name="Topic Relevance",
            engine="openpyxl",
        )

        hierarchy = pd.read_excel(
            file,
            sheet_name="Hierarchy Evaluation",
            engine="openpyxl",
        )

        annotator = detect_annotator(topic)

        topic_parts[annotator] = topic
        hierarchy_parts[annotator] = hierarchy

        print(file.name, "->", annotator)

    topic_keys = [
        "Query_ID",
        "Query",
        "Anonymous_System",
        "Level",
        "Rank",
        "Topic",
    ]

    hierarchy_keys = [
        "Query_ID",
        "Query",
        "Anonymous_System",
    ]

    topic = (
        topic_parts["A1"][topic_keys]
        .copy()
        .reset_index(drop=True)
    )

    hierarchy = (
        hierarchy_parts["A1"][hierarchy_keys]
        .copy()
        .reset_index(drop=True)
    )

    for a in ANNOTATORS:

        topic[f"{a}_Relevance"] = pd.to_numeric(
            topic_parts[a][f"{a}_Relevance"],
            errors="coerce",
        ).reset_index(drop=True)

        for criterion in CRITERIA:

            col = f"{a}_{criterion}"

            hierarchy[col] = pd.to_numeric(
                hierarchy_parts[a][col],
                errors="coerce",
            ).reset_index(drop=True)

    return topic, hierarchy


def fleiss_for_matrix(ratings):

    ratings = np.asarray(ratings, dtype=float)

    ratings = ratings[
        ~np.isnan(ratings).any(axis=1)
    ]

    if len(ratings) == 0:
        return np.nan, 0

    table = np.zeros(
        (len(ratings), 4),
        dtype=int,
    )

    for i, row in enumerate(ratings.astype(int)):

        for score in range(4):

            table[i, score] = np.sum(
                row == score
            )

    kappa = fleiss_kappa(table)

    return float(kappa), len(ratings)


def krippendorff_ordinal(ratings):

    ratings = np.asarray(
        ratings,
        dtype=float,
    ).T

    valid = ~np.isnan(
        ratings
    ).all(axis=0)

    ratings = ratings[:, valid]

    if ratings.shape[1] == 0:
        return np.nan, 0

    alpha = krippendorff.alpha(
        reliability_data=ratings,
        level_of_measurement="ordinal",
    )

    return float(alpha), ratings.shape[1]


def agreement_analysis(topic, hierarchy):

    fleiss_results = []
    alpha_results = []

    relevance_columns = [
        f"{a}_Relevance"
        for a in ANNOTATORS
    ]

    matrix = topic[
        relevance_columns
    ].to_numpy()

    kappa, n = fleiss_for_matrix(matrix)

    alpha, n_alpha = krippendorff_ordinal(
        matrix
    )

    fleiss_results.append({
        "Measure": "Topic Relevance",
        "Fleiss_Kappa": kappa,
        "N_Items": n,
    })

    alpha_results.append({
        "Measure": "Topic Relevance",
        "Krippendorff_Alpha_Ordinal": alpha,
        "N_Items": n_alpha,
    })

    pooled = []

    for criterion in CRITERIA:

        cols = [
            f"{a}_{criterion}"
            for a in ANNOTATORS
        ]

        matrix = hierarchy[
            cols
        ].to_numpy()

        kappa, n = fleiss_for_matrix(
            matrix
        )

        alpha, n_alpha = (
            krippendorff_ordinal(
                matrix
            )
        )

        label = criterion.replace(
            "_",
            " ",
        )

        fleiss_results.append({
            "Measure": label,
            "Fleiss_Kappa": kappa,
            "N_Items": n,
        })

        alpha_results.append({
            "Measure": label,
            "Krippendorff_Alpha_Ordinal": alpha,
            "N_Items": n_alpha,
        })

        pooled.append(matrix)

    pooled_matrix = np.vstack(
        pooled
    )

    kappa, n = fleiss_for_matrix(
        pooled_matrix
    )

    alpha, n_alpha = (
        krippendorff_ordinal(
            pooled_matrix
        )
    )

    fleiss_results.append({
        "Measure": "Hierarchy Pooled",
        "Fleiss_Kappa": kappa,
        "N_Items": n,
    })

    alpha_results.append({
        "Measure": "Hierarchy Pooled",
        "Krippendorff_Alpha_Ordinal": alpha,
        "N_Items": n_alpha,
    })

    return (
        pd.DataFrame(fleiss_results),
        pd.DataFrame(alpha_results),
    )


def relevance_long(topic):

    all_rows = []

    for a in ANNOTATORS:

        col = f"{a}_Relevance"

        temp = topic[
            [
                "Query_ID",
                "Anonymous_System",
                col,
            ]
        ].copy()

        temp["Annotator"] = a

        temp["Rating"] = pd.to_numeric(
            temp[col],
            errors="coerce",
        )

        result = (
            temp.groupby(
                [
                    "Query_ID",
                    "Anonymous_System",
                    "Annotator",
                ],
                as_index=False,
            )["Rating"]
            .mean()
            .rename(
                columns={
                    "Rating":
                    "Mean_Relevance"
                }
            )
        )

        all_rows.append(result)

    return pd.concat(
        all_rows,
        ignore_index=True,
    )


def dcg(scores, k=8):

    scores = np.asarray(
        scores,
        dtype=float,
    )[:k]

    if len(scores) == 0:
        return np.nan

    gains = (
        2.0 ** scores
    ) - 1.0

    discounts = np.log2(
        np.arange(
            2,
            len(scores) + 2,
        )
    )

    return float(
        np.sum(
            gains / discounts
        )
    )


def ndcg_long(topic):

    rows = []

    for a in ANNOTATORS:

        col = f"{a}_Relevance"

        temp = topic[
            [
                "Query_ID",
                "Anonymous_System",
                "Level",
                "Rank",
                col,
            ]
        ].copy()

        temp["Rating"] = pd.to_numeric(
            temp[col],
            errors="coerce",
        )

        temp["_level"] = (
            temp["Level"]
            .astype(str)
            .str.lower()
            .map(LEVEL_ORDER)
            .fillna(99)
        )

        temp["_rank"] = pd.to_numeric(
            temp["Rank"],
            errors="coerce",
        ).fillna(999999)

        grouped = temp.groupby(
            [
                "Query_ID",
                "Anonymous_System",
            ]
        )

        for (qid, system), group in grouped:

            group = (
                group
                .dropna(
                    subset=["Rating"]
                )
                .sort_values(
                    ["_level", "_rank"]
                )
            )

            ratings = (
                group["Rating"]
                .to_numpy(
                    dtype=float
                )
            )

            if len(ratings) == 0:

                value = np.nan

            else:

                actual = dcg(
                    ratings,
                    8,
                )

                ideal = dcg(
                    np.sort(
                        ratings
                    )[::-1],
                    8,
                )

                if ideal == 0:
                    value = 0.0
                else:
                    value = (
                        actual / ideal
                    )

            rows.append({
                "Query_ID": qid,
                "Anonymous_System": system,
                "Annotator": a,
                "NDCG@8": value,
            })

    return pd.DataFrame(rows)


def hierarchy_long(hierarchy):

    rows = []

    for a in ANNOTATORS:

        temp = hierarchy[
            [
                "Query_ID",
                "Anonymous_System",
            ]
        ].copy()

        temp["Annotator"] = a

        for criterion in CRITERIA:

            temp[criterion] = (
                pd.to_numeric(
                    hierarchy[
                        f"{a}_{criterion}"
                    ],
                    errors="coerce",
                )
            )

        temp[
            "Hierarchy_Overall"
        ] = temp[
            CRITERIA
        ].mean(axis=1)

        rows.append(temp)

    return pd.concat(
        rows,
        ignore_index=True,
    )


def query_scores(df, metric):

    return (
        df.groupby(
            [
                "Query_ID",
                "Anonymous_System",
            ],
            as_index=False,
        )[metric]
        .mean()
    )


def wilcoxon_comparison(
    query_df,
    metric,
):

    results = []

    target = (
        query_df[
            query_df[
                "Anonymous_System"
            ] == "System H"
        ][
            [
                "Query_ID",
                metric,
            ]
        ]
        .rename(
            columns={
                metric: "LLM_QTH"
            }
        )
    )

    for system in [
        "System A",
        "System B",
        "System C",
        "System D",
        "System E",
        "System F",
        "System G",
    ]:

        baseline = (
            query_df[
                query_df[
                    "Anonymous_System"
                ] == system
            ][
                [
                    "Query_ID",
                    metric,
                ]
            ]
            .rename(
                columns={
                    metric: "Baseline"
                }
            )
        )

        merged = (
            target
            .merge(
                baseline,
                on="Query_ID",
                how="inner",
            )
            .dropna()
        )

        difference = (
            merged["LLM_QTH"]
            - merged["Baseline"]
        )

        if len(merged) == 0:

            statistic = np.nan
            p_value = np.nan

        elif np.allclose(
            difference.to_numpy(),
            0,
        ):

            statistic = 0.0
            p_value = 1.0

        else:

            statistic, p_value = (
                wilcoxon(
                    merged[
                        "LLM_QTH"
                    ],
                    merged[
                        "Baseline"
                    ],
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
            )

        results.append({
            "Metric": metric,
            "LLM_QTH_System":
                "System H",
            "Baseline_System":
                system,
            "Baseline_Model":
                MODEL_MAP[system],
            "N_Paired_Queries":
                len(merged),
            "LLM_QTH_Mean":
                merged[
                    "LLM_QTH"
                ].mean(),
            "Baseline_Mean":
                merged[
                    "Baseline"
                ].mean(),
            "Mean_Difference":
                difference.mean(),
            "Wilcoxon_Statistic":
                statistic,
            "Raw_p":
                p_value,
        })

    result_df = pd.DataFrame(
        results
    )

    valid = result_df[
        "Raw_p"
    ].notna()

    if valid.any():

        reject, adjusted, _, _ = (
            multipletests(
                result_df.loc[
                    valid,
                    "Raw_p",
                ].to_numpy(),
                alpha=0.05,
                method="holm",
            )
        )

        result_df.loc[
            valid,
            "Holm_p",
        ] = adjusted

        result_df.loc[
            valid,
            "Significant_0.05",
        ] = reject

    return result_df


def pairwise_analysis(
    relevance,
    ndcg,
    hierarchy,
):

    all_results = []

    rel_query = query_scores(
        relevance,
        "Mean_Relevance",
    )

    all_results.append(
        wilcoxon_comparison(
            rel_query,
            "Mean_Relevance",
        )
    )

    ndcg_query = query_scores(
        ndcg,
        "NDCG@8",
    )

    all_results.append(
        wilcoxon_comparison(
            ndcg_query,
            "NDCG@8",
        )
    )

    metrics = (
        CRITERIA
        + ["Hierarchy_Overall"]
    )

    for metric in metrics:

        hierarchy_query = (
            query_scores(
                hierarchy,
                metric,
            )
        )

        all_results.append(
            wilcoxon_comparison(
                hierarchy_query,
                metric,
            )
        )

    return pd.concat(
        all_results,
        ignore_index=True,
    )


def system_scores(
    relevance,
    hierarchy,
    keep_annotators,
):

    rel = relevance[
        relevance[
            "Annotator"
        ].isin(
            keep_annotators
        )
    ]

    hier = hierarchy[
        hierarchy[
            "Annotator"
        ].isin(
            keep_annotators
        )
    ]

    rel_scores = (
        rel.groupby(
            "Anonymous_System"
        )[
            "Mean_Relevance"
        ]
        .mean()
    )

    hierarchy_scores = (
        hier.groupby(
            "Anonymous_System"
        )[
            "Hierarchy_Overall"
        ]
        .mean()
    )

    systems = sorted(
        set(
            rel_scores.index
        )
        |
        set(
            hierarchy_scores.index
        )
    )

    result = pd.DataFrame({
        "Anonymous_System":
            systems
    })

    result[
        "Mean_Relevance"
    ] = result[
        "Anonymous_System"
    ].map(rel_scores)

    result[
        "Hierarchy_Overall"
    ] = result[
        "Anonymous_System"
    ].map(
        hierarchy_scores
    )

    result["Model"] = (
        result[
            "Anonymous_System"
        ].map(MODEL_MAP)
    )

    return result


def leave_one_out(
    relevance,
    hierarchy,
):

    full = system_scores(
        relevance,
        hierarchy,
        ANNOTATORS,
    )

    loo_rows = []
    stability_rows = []

    for omitted in ANNOTATORS:

        keep = [
            a for a in ANNOTATORS
            if a != omitted
        ]

        scores = system_scores(
            relevance,
            hierarchy,
            keep,
        )

        scores[
            "Omitted_Annotator"
        ] = omitted

        loo_rows.append(scores)

        for metric in [
            "Mean_Relevance",
            "Hierarchy_Overall",
        ]:

            full_metric = (
                full[
                    [
                        "Anonymous_System",
                        metric,
                    ]
                ]
                .rename(
                    columns={
                        metric: "Full"
                    }
                )
            )

            loo_metric = (
                scores[
                    [
                        "Anonymous_System",
                        metric,
                    ]
                ]
                .rename(
                    columns={
                        metric: "LOO"
                    }
                )
            )

            merged = (
                full_metric
                .merge(
                    loo_metric,
                    on="Anonymous_System",
                )
                .sort_values(
                    "Anonymous_System"
                )
                .reset_index(
                    drop=True
                )
            )

            rho, p_value = (
                spearmanr(
                    merged["Full"],
                    merged["LOO"],
                )
            )

            full_rank = (
                merged["Full"]
                .rank(
                    ascending=False,
                    method="min",
                )
            )

            loo_rank = (
                merged["LOO"]
                .rank(
                    ascending=False,
                    method="min",
                )
            )

            full_top = (
                merged.loc[
                    merged[
                        "Full"
                    ].idxmax(),
                    "Anonymous_System",
                ]
            )

            loo_top = (
                merged.loc[
                    merged[
                        "LOO"
                    ].idxmax(),
                    "Anonymous_System",
                ]
            )

            stability_rows.append({
                "Omitted_Annotator":
                    omitted,
                "Metric":
                    metric,
                "Spearman_rho":
                    rho,
                "Spearman_p":
                    p_value,
                "Full_Top_System":
                    full_top,
                "LOO_Top_System":
                    loo_top,
                "Rank_Ordering_Unchanged":
                    bool(
                        np.array_equal(
                            full_rank.to_numpy(),
                            loo_rank.to_numpy(),
                        )
                    ),
            })

    loo_df = pd.concat(
        loo_rows,
        ignore_index=True,
    )

    loo_df = loo_df[
        [
            "Omitted_Annotator",
            "Anonymous_System",
            "Model",
            "Mean_Relevance",
            "Hierarchy_Overall",
        ]
    ]

    stability_df = pd.DataFrame(
        stability_rows
    )

    return loo_df, stability_df


def main():

    folder = Path(".")

    output_folder = (
        folder
        / "additional_human_evaluation"
    )

    output_folder.mkdir(
        exist_ok=True
    )

    print(
        "\nReading five annotator workbooks..."
    )

    topic, hierarchy = load_data(
        folder
    )

    print(
        "\nTopic rows:",
        len(topic),
    )

    print(
        "Hierarchy rows:",
        len(hierarchy),
    )

    relevance = relevance_long(
        topic
    )

    ndcg = ndcg_long(
        topic
    )

    hierarchy_scores = (
        hierarchy_long(
            hierarchy
        )
    )

    print(
        "\nComputing inter-annotator agreement..."
    )

    fleiss_df, alpha_df = (
        agreement_analysis(
            topic,
            hierarchy,
        )
    )

    print(
        "Computing pairwise significance..."
    )

    pairwise_df = (
        pairwise_analysis(
            relevance,
            ndcg,
            hierarchy_scores,
        )
    )

    print(
        "Computing leave-one-annotator-out robustness..."
    )

    loo_df, stability_df = (
        leave_one_out(
            relevance,
            hierarchy_scores,
        )
    )

    fleiss_df.to_csv(
        output_folder
        / "Fleiss_Kappa.csv",
        index=False,
    )

    alpha_df.to_csv(
        output_folder
        / "Krippendorff_Alpha.csv",
        index=False,
    )

    pairwise_df.to_csv(
        output_folder
        / "Pairwise_Significance.csv",
        index=False,
    )

    loo_df.to_csv(
        output_folder
        / "Leave_One_Annotator_Out.csv",
        index=False,
    )

    stability_df.to_csv(
        output_folder
        / "Leave_One_Annotator_Out_Rank_Stability.csv",
        index=False,
    )

    excel_file = (
        output_folder
        / "Additional_Human_Evaluation.xlsx"
    )

    with pd.ExcelWriter(
        excel_file,
        engine="openpyxl",
    ) as writer:

        fleiss_df.to_excel(
            writer,
            sheet_name="Fleiss Kappa",
            index=False,
        )

        alpha_df.to_excel(
            writer,
            sheet_name="Krippendorff Alpha",
            index=False,
        )

        pairwise_df.to_excel(
            writer,
            sheet_name="Pairwise Significance",
            index=False,
        )

        loo_df.to_excel(
            writer,
            sheet_name="LOO Metrics",
            index=False,
        )

        stability_df.to_excel(
            writer,
            sheet_name="LOO Rank Stability",
            index=False,
        )

    print("\nSUCCESS")
    print(
        "Created folder:",
        output_folder
    )

    print(
        "\nFiles created:"
    )

    for file in sorted(
        output_folder.iterdir()
    ):
        print(" -", file.name)


if __name__ == "__main__":
    main()