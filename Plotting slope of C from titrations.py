import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "figure.titlesize": 20,
})




results_df = pd.read_csv("titration_plateau_results_only_full_titrations_titrant_mass.csv")
results_df = results_df[results_df["file_not_good"]!="not_good"]
results_df = results_df[results_df["sample type"]!= "Nuts"]
# Convert date to datetime if needed
results_df["date"] = pd.to_datetime(results_df["date"],dayfirst=True).dt.strftime("%d-%m")
#%%

# -----------------------------------------------
# Unique sample types
# -----------------------------------------------
sample_types = results_df["sample type"].unique()

# -----------------------------------------------
# Colors and markers (up to 5 categories)
# -----------------------------------------------
palette = sns.color_palette("tab10", len(sample_types))
markers = ["o", "s", "D", "^", "v"]  # circle, square, diamond, triangle_up, triangle_down

type_color = {stype: palette[i] for i, stype in enumerate(sample_types)}
type_marker = {stype: markers[i] for i, stype in enumerate(sample_types)}

# -----------------------------------------------
# Plot
# -----------------------------------------------
plt.figure()

for stype in sample_types:
    subset = results_df[results_df["sample type"] == stype]
    plt.scatter(
        subset["Alkalinity"],
        subset["slope"],
        s=80,
        alpha=0.75,
        color=type_color[stype],
        marker=type_marker[stype],
        label=stype
    )

plt.xlabel("Alkalinity (µmol/kg)")
plt.ylabel(r"Slope/($\frac{\Delta DIC (\%)}{acid (ml)}$)")
plt.legend(title="Sample Type",fontsize = 14)
plt.tight_layout()
plt.show()


#correcting for titrant molinity 
plt.figure()
for stype in sample_types:
    subset = results_df[results_df["sample type"] == stype]
    plt.scatter(
        subset["Alkalinity"],
        subset["slope"]/subset["Titrant Molinity"],
        s=80,
        alpha=0.75,
        color=type_color[stype],
        marker=type_marker[stype],
        label=stype
    )

plt.xlabel("Alkalinity (µmol/kg)")
plt.ylabel(r"Slope/($\frac{\Delta DIC (\%)}{acid (mM)}$)")
plt.legend(title="Sample Type",fontsize = 14)
plt.tight_layout()
plt.show()
#%%
# ----- FIGURE 1: Slope vs Date -----
plt.figure(figsize=(10,5))
plt.scatter(results_df["date"], results_df["slope"])
plt.xlabel("Date")
plt.ylabel("Slope of early C%")
plt.title("Slope of Early C% vs Date")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# ----- FIGURE 1: Slope vs Date -----
plt.figure(figsize=(10,5))
plt.scatter(results_df["sample type"], results_df["slope"])
plt.xlabel("sample type")
plt.ylabel("Slope of early C%")
plt.title("Slope of Early C% vs Sample Type")
plt.tight_layout()
plt.show()
#%%
plt.figure(figsize=(10,5))
plt.scatter(results_df[results_df["sample type"]=="Junk"]["Alkalinity"], results_df[results_df["sample type"]=="Junk"]["slope"],label = "Junk")
plt.scatter(results_df[results_df["sample type"]=="Nose"]["Alkalinity"], results_df[results_df["sample type"]=="Nose"]["slope"],label = "Nose")
plt.scatter(results_df[results_df["sample type"]=="Junk_old"]["Alkalinity"], results_df[results_df["sample type"]=="Junk_old"]["slope"],label = "Old junk")
plt.scatter(results_df[results_df["sample type"]=="Junk_old_2"]["Alkalinity"], results_df[results_df["sample type"]=="Junk_old_2"]["slope"],label = "Old junk 2")
plt.scatter(results_df[results_df["sample type"]=="Junk_old_3"]["Alkalinity"], results_df[results_df["sample type"]=="Junk_old_3"]["slope"],label = "Old junk 3")
plt.xlabel("Alkalinity")
plt.ylabel("Slope of early C%")
plt.legend()
plt.tight_layout()
plt.show()
#%%
# ----- FIGURE 2: R^2 vs Date -----
plt.figure(figsize=(10,5))
plt.scatter(results_df["date"], results_df["r_squared"])
plt.plot(results_df["date"], results_df["r_squared"], alpha=0.4)
plt.xlabel("Date")
plt.ylabel("R² of linear fit")
plt.title("Linear Fit R² vs Date")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# ----- FIGURE 3: Plateau C% vs Date -----
plt.figure(figsize=(10,5))
plt.scatter(results_df["date"], results_df["plateau_C_percent"])
plt.plot(results_df["date"], results_df["plateau_C_percent"], alpha=0.4)
plt.xlabel("Date")
plt.ylabel("Plateau C%")
plt.title("Plateau Level of Theoretical C% vs Date")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# ----- FIGURE 4: Slope vs Reference DIC -----
plt.figure(figsize=(8,6))
plt.scatter(results_df["reference_DIC"], results_df["slope"])
plt.xlabel("Reference DIC (µmol/kg)")
plt.ylabel("Slope")
plt.title("Slope vs Reference DIC")
plt.tight_layout()
plt.show()

