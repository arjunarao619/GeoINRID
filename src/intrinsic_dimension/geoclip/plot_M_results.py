import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ——— Data ———
df = pd.DataFrame({
    'Hierarchy Level M': [3, 4, 5, 6, 7, 8, 9, 10],
    'Intrinsic Dimension (MLE)': [15.21, 8.58, 7.88, 7.43, 7.10, 6.99, 6.68, 6.61]
})

# ——— Seaborn styling for camera-ready ———
sns.set_theme(style="whitegrid", font_scale=1.2)

# ——— Plot ———
plt.figure(figsize=(6, 4), dpi=300)
ax = sns.barplot(
    x='Hierarchy Level M',
    y='Intrinsic Dimension (MLE)',
    data=df,
    edgecolor=".2"
)

# ——— Labels & Title ———
ax.set_xlabel("Hierarchy Level $M$", fontsize=12)
ax.set_ylabel("Intrinsic Dimension (MLE)", fontsize=12)
ax.set_title("Intrinsic Dimension vs. RFF Hierarchy Level", fontsize=14)

# ——— Grid & Layout ———
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()

# ——— Save & Show ———
plt.savefig("id_vs_hierarchy_seaborn.png", dpi=300)
