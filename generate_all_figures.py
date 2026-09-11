import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# OUTPUT FOLDER
# ============================================================

os.makedirs("paper_figures", exist_ok=True)

# ============================================================
# FINAL HUMAN-EVALUATION RESULTS
# ============================================================

models = [
    "LDA",
    "FLAN-T5-v2",
    "FLAN-T5-v3",
    "FLAN-T5-v1",
    "KeyBERT",
    "BERTopic",
    "NMF",
    "LLM-QTH"
]

mean_relevance = np.array([
    0.8339,
    0.8276,
    0.8636,
    0.8391,
    1.3401,
    0.8687,
    0.8877,
    2.0761
])

relevance_sd = np.array([
    0.9966,
    1.0152,
    0.9880,
    1.0192,
    0.9008,
    0.9963,
    0.9763,
    0.7676
])

ndcg = np.array([
    0.3601,
    0.3863,
    0.4750,
    0.3895,
    0.5934,
    0.3360,
    0.4288,
    0.8005
])

ndcg_sd = np.array([
    0.3720,
    0.4139,
    0.4183,
    0.4143,
    0.3341,
    0.3566,
    0.3706,
    0.1957
])

coherence = np.array([
    1.084,
    0.828,
    1.300,
    1.184,
    1.396,
    1.136,
    1.364,
    2.172
])

level_appropriateness = np.array([
    1.260,
    1.008,
    1.464,
    1.212,
    1.400,
    1.232,
    1.224,
    1.972
])

progression = np.array([
    1.436,
    1.408,
    1.468,
    1.384,
    1.772,
    1.432,
    1.592,
    2.292
])

non_redundancy = np.array([
    1.632,
    1.204,
    1.320,
    1.204,
    1.960,
    1.460,
    1.824,
    2.360
])

x = np.arange(len(models))

# ============================================================
# FIGURE X
# RELEVANCE + NDCG@8
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5.5)
)

axes[0].bar(
    x,
    mean_relevance,
    yerr=relevance_sd,
    capsize=4
)

axes[0].set_xticks(x)
axes[0].set_xticklabels(
    models,
    rotation=35,
    ha="right"
)

axes[0].set_ylabel("Mean relevance")
axes[0].set_title("(a) Topic relevance")
axes[0].set_ylim(bottom=0)

axes[1].bar(
    x,
    ndcg,
    yerr=ndcg_sd,
    capsize=4
)

axes[1].set_xticks(x)
axes[1].set_xticklabels(
    models,
    rotation=35,
    ha="right"
)

axes[1].set_ylabel("Within-output NDCG@8")
axes[1].set_title("(b) Ranking quality")
axes[1].set_ylim(0, 1.05)

plt.tight_layout()

plt.savefig(
    "paper_figures/FigX_Relevance_NDCG.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "paper_figures/FigX_Relevance_NDCG.pdf",
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FIGURE Y
# SPIDER / RADAR CHART
# ============================================================

criteria = [
    "Coherence",
    "Level\nAppropriateness",
    "Progression",
    "Non-\nRedundancy"
]

data = np.column_stack([
    coherence,
    level_appropriateness,
    progression,
    non_redundancy
])

angles = np.linspace(
    0,
    2 * np.pi,
    len(criteria),
    endpoint=False
).tolist()

angles += angles[:1]

fig = plt.figure(
    figsize=(9, 8)
)

ax = fig.add_subplot(
    111,
    polar=True
)

for i, model in enumerate(models):

    values = data[i].tolist()
    values += values[:1]

    ax.plot(
        angles,
        values,
        linewidth=1.6,
        label=model
    )

    ax.fill(
        angles,
        values,
        alpha=0.03
    )

ax.set_xticks(
    angles[:-1]
)

ax.set_xticklabels(
    criteria
)

ax.set_ylim(
    0,
    3
)

ax.set_yticks([
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0
])

ax.set_title(
    "Human evaluation of hierarchy quality",
    pad=25
)

ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.05, 1.10),
    frameon=False
)

plt.tight_layout()

plt.savefig(
    "paper_figures/FigY_Hierarchy_Spider.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "paper_figures/FigY_Hierarchy_Spider.pdf",
    bbox_inches="tight"
)

plt.close()

# ============================================================
# DONE
# ============================================================

print("All paper figures generated successfully.")
print("Open the folder: paper_figures")