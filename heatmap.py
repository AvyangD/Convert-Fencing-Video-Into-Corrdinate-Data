import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("fencing_data2.csv")

coord_df = df[[col for col in df.columns if col.startswith(("x_", "y_", "z_"))]]

corr = coord_df.corr()

plt.figure(figsize=(22, 18))
sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    cbar=True,
    square=True,
    xticklabels=True,
    yticklabels=True
)
plt.title("Correlation Heatmap of Fencing Coordinates (X, Y, Z only)", fontsize=18)
plt.xticks(rotation=90, fontsize=7)  
plt.yticks(rotation=0, fontsize=7)   
plt.tight_layout()

plt.savefig("heatmap_coords_labeled.png", dpi=300)
plt.show()