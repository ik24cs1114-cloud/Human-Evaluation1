import os
import numpy as np
import matplotlib.pyplot as plt

# Create figures folder
os.makedirs("figures", exist_ok=True)

metrics = [
    "Topic\nrelevance",
    "Coherence",
    "Level\nappropriateness",
    "Progression",
    "Non-redundancy"
]

mmlu = [2.437, 2.733, 2.533, 2.720, 2.413]
medmcqa = [2.492, 2.653, 2.693, 2.573, 2.467]

x = np.arange(len(metrics))
width = 0.34

fig, ax = plt.subplots(figsize=(8.5, 4.8))

bars1 = ax.bar(
    x - width / 2,
    mmlu,
    width,
    label="MMLU",
    edgecolor="black",
    linewidth=0.6
)

bars2 = ax.bar(
    x + width / 2,
    medmcqa,
    width,
    label="MedMCQA",
    edgecolor="black",
    linewidth=0.6
)

ax.set_ylabel("Mean human rating (0-3)", fontsize=11)
ax.set_xlabel("Evaluation criterion", fontsize=11)

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=9)
ax.set_ylim(0, 3.1)

ax.legend(fontsize=9, loc="upper right")

# Add numerical values above bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.035,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    "figures/cross_domain_human_evaluation.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "figures/cross_domain_human_evaluation.pdf",
    bbox_inches="tight"
)

plt.show()

print("Figure created successfully.")
print("Saved in: figures/cross_domain_human_evaluation.png")
print("Saved in: figures/cross_domain_human_evaluation.pdf")