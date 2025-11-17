import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------
# Load your dataframe
# --------------------------------------------------------
df = pd.read_csv("DIC_logbook.csv")

# Ensure proper datetime formatting
df["File date"] = pd.to_datetime(df["File date"], dayfirst=True)

# --------------------------------------------------------
# 1. FILTER REFERENCES ONLY
# --------------------------------------------------------
ref_df = df[df["Reference"] == 1].copy()

# --------------------------------------------------------
# 2. OPTIONAL: Extract just the date (no timestamp)
# --------------------------------------------------------
ref_df["Day"] = ref_df["File date"].dt.date

# --------------------------------------------------------
# 3. GROUP FOR STATISTICS
# --------------------------------------------------------
# Get mean + std for each day and Sample Type combination
stats = (
    ref_df.groupby(["Day", "Sample type"])["Negative removed DIC (umol/L)"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

print(stats)
sample_types = ref_df["Sample type"].unique()

# Define markers and colors (repeat if necessary)
markers = ["o", "s", "^", "D", "v", "P"]
colors = plt.cm.tab10(range(len(sample_types)))

marker_map = {stype: markers[i % len(markers)] for i, stype in enumerate(sample_types)}
color_map  = {stype: colors[i % len(colors)]  for i, stype in enumerate(sample_types)}

plt.figure(figsize=(12,6))

# Loop through sample types
for stype in sample_types:
    stype_data = stats[stats["Sample type"] == stype]

    plt.errorbar(
        stype_data["Day"],
        stype_data["mean"],
        yerr=stype_data["std"],
        fmt=marker_map[stype],
        color=color_map[stype],
        label=f"{stype}",
        capsize=4,
        linestyle="none"
    )

plt.xlabel("Day")
plt.ylabel("Negative removed DIC (µmol/L)")
plt.title("Daily Reference DIC (Reference = 1)")
plt.legend(title="Sample Type")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
