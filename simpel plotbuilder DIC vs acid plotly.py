import pandas as pd
import plotly.express as px

# Read Excel file
excel_file = "logbook_automated_by_python_testing.xlsx"
df = pd.read_excel(excel_file)

# Make sure columns exist
required_cols = ["Calculated DIC (umol/kg)", "acid added (mL)", "date", "waiting time (minutes)", "Reference DIC (umol/kg)", "batch" ]
if not all(col in df.columns for col in required_cols):
    raise ValueError("Required columns missing in the Excel file.")

# Drop rows with missing data in the columns we need
plot_df = df.dropna(subset=required_cols).copy()

# Filter: only rows with short waiting time
plot_df = plot_df[plot_df["waiting time (minutes)"] <= 0.5]

# Select a specific date or batch
#plot_df = plot_df[plot_df["date"] == "23-Oct"]
plot_df = plot_df[plot_df["date"] != "9-Oct"]
#plot_df = plot_df[plot_df["batch"] == 1]

# Convert to numeric where needed
plot_df["acid added (mL)"] = pd.to_numeric(plot_df["acid added (mL)"], errors="coerce")
plot_df["Calculated DIC (umol/kg)"] = pd.to_numeric(plot_df["Calculated DIC (umol/kg)"], errors="coerce")
plot_df["Reference DIC (umol/kg)"] = pd.to_numeric(plot_df["Reference DIC (umol/kg)"], errors="coerce")
plot_df["waiting time (minutes)"] = pd.to_numeric(plot_df["waiting time (minutes)"], errors="coerce")
plot_df["Titration duration (seconds)"] = pd.to_numeric(plot_df["Titration duration (seconds)"], errors="coerce")

# Compute percentage DIC
plot_df["Percentage DIC"] = 100 * plot_df["Calculated DIC (umol/kg)"] / plot_df["Reference DIC (umol/kg)"]

# Create an interactive scatter plot
fig = px.scatter(
    plot_df,
    x="acid added (mL)",
    y="Percentage DIC",
    color="Titration duration (seconds)",      # continuous color scale
    symbol="date",                       # different marker shapes for different days
    size_max=12,
    hover_data={
        "acid added (mL)": True,
        "Percentage DIC": True,
        "waiting time (minutes)": True,
        "date": True,
        "batch": True
    },
    title="DIC vs Acid Added — Colored by Titration duration, Symbol by Date",
    color_continuous_scale="Viridis"
)

# Improve layout
fig.update_layout(
    xaxis_title="Acid Added (mL)",
    yaxis_title="Remaining DIC (%)",
    legend_title_text="Waiting Time (min)",
    template="plotly_white",
    title_font=dict(size=20),
    font=dict(size=16)
)

fig.show()

