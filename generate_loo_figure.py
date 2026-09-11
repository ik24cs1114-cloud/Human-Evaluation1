import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Input file
input_file = Path(
    "additional_human_evaluation"
) / "Leave_One_Annotator_Out.csv"

# Output folder
output_folder = Path("paper_figures")
output_folder.mkdir(exist_ok=True)

# Read data
df = pd.read_csv(input_file)

# Correct system names
model_map = {
    "System A": "LDA",
    "System B": "FLAN-T5-v2",
    "System C": "FLAN-T5-v3",
    "System D": "FLAN-T5-v1",
    "System E": "KeyBERT",
    "System F": "BERTopic",
    "System G": "NMF",
    "System H": "LLM-QTH",
}

# Order of omitted annotators
annotator_order = ["A1", "A2", "A3", "A4", "A5"]

fig, ax = plt.subplots(figsize=(9, 5.5))

for system in model_map:

    temp = df[
        df["Anonymous_System"] == system
    ].copy()

    temp["Omitted_Annotator"] = pd.Categorical(
        temp["Omitted_Annotator"],
        categories=annotator_order,
        ordered=True
    )

    temp = temp.sort_values("Omitted_Annotator")

    ax.plot(
        temp["Omitted_Annotator"],
        temp["Mean_Relevance"],
        marker="o",
        linewidth=1.8,
        label=model_map[system]
    )

ax.set_xlabel("Omitted Annotator")
ax.set_ylabel("Mean Relevance")
ax.set_title("Leave-One-Annotator-Out Robustness Analysis")

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.35
)

ax.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=False
)

fig.tight_layout()

# Save high-resolution PNG
png_file = (
    output_folder /
    "Fig_LOO_Annotator_Robustness.png"
)

fig.savefig(
    png_file,
    dpi=600,
    bbox_inches="tight"
)

# Save PDF for paper
pdf_file = (
    output_folder /
    "Fig_LOO_Annotator_Robustness.pdf"
)

fig.savefig(
    pdf_file,
    bbox_inches="tight"
)

plt.close()

print("LOO figure generated successfully.")
print("PNG:", png_file)
print("PDF:", pdf_file)
