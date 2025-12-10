
"""
Environmental Data Cleaning & Moving Average Smoothing
------------------------------------------------
Purpose: Load and clean multiple EasyLog sensor recordings, merge them into a single time-aligned dataset, and apply moving-average smoothing.
Output: Smoothed and merged Temperature, Humidity, and Dew Point time series

Author: Mohamed Benouis

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

OUTPUT_FIG_DIR = "figures"
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)
# Data

FILES = [ "EasyLog USB 1_04aug2025_13h33.txt", "EasyLog USB 1_12aug2025_14h01.txt", "EasyLog USB 1_19aug2025_12h16.txt"]

# 1) DELIMITER DETECTOR: Check the delimiter (, ; \t etc.) is used in a text file

def detect_delimiter(file_path):
    """Automatically detect delimiter in a text file."""
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            if line.strip():
                try:
                    return csv.Sniffer().sniff(line).delimiter
                except:
                    return ";"
    return ";"


# 2) LOAD & CLEAN Three FILES ###############################################

print("\n1) Reading and merging txt files...")
data_frames = []

for file in FILES:
    if not os.path.exists(file):
        print(f" ***File not found: {file}")
        continue

    delimiter = detect_delimiter(file)
    print(f"\n### Detected delimiter for {file}: '{delimiter}'")

    df = pd.read_csv(file, encoding="latin-1", delimiter=delimiter)
    df.columns = [c.strip() for c in df.columns]

    # Convert types
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"])

    df["Temperature"] = pd.to_numeric(df["Celsius(°C)"], errors="coerce")
    df["Humidity"]    = pd.to_numeric(df["Humidity(%rh)"], errors="coerce")
    df["Dewpoint"]    = pd.to_numeric(df["Dew Point(°C)"], errors="coerce")

    df = df[["Time","Temperature","Humidity","Dewpoint"]]
    df = df.set_index("Time").sort_index()

    data_frames.append(df)

    print(f"{file} — Valid samples: {len(df)}")

# Combine all files
df_combined = pd.concat(data_frames).sort_index()

# 3) SUMMARY STATISTICS #####################


import numpy as np

print("\n2) Generating summary statistics...\n")

summary = {}

def compute_stats(df):
    """Return duration, count, and mean ± std for each variable."""
    duration_days = (df.index.max() - df.index.min()).total_seconds() / 86400

    return {
        "rows": len(df),
        "duration_days": duration_days,
        "temp_mean": df["Temperature"].mean(),
        "temp_std": df["Temperature"].std(),
        "hum_mean": df["Humidity"].mean(),
        "hum_std": df["Humidity"].std(),
        "dew_mean": df["Dewpoint"].mean(),
        "dew_std": df["Dewpoint"].std(),
    }

# Per-file stats
for file, df in zip(FILES, data_frames):
    summary[file] = compute_stats(df)

# Overall stats
summary["Overall"] = compute_stats(df_combined)

# Build a table-like printout
print("File Summary:\n")
for key, stats in summary.items():
    print(f"=== {key} ===")
    print(f"Raw/Clean Rows     : {stats['rows']}")
    print(f"Duration           : {stats['duration_days']:.2f} days")
    print(f"Temperature (°C)   : {stats['temp_mean']:.2f} ± {stats['temp_std']:.2f}")
    print(f"Humidity (%rh)     : {stats['hum_mean']:.2f} ± {stats['hum_std']:.2f}")
    print(f"Dew Point (°C)     : {stats['dew_mean']:.2f} ± {stats['dew_std']:.2f}")
    print()

# Sampling interval
median_interval_sec = (
    df_combined.index.to_series()
    .diff()
    .dropna()
    .median()
    .total_seconds()
)

#######sampling frequency #####################
duration_hours = (df_combined.index.max() - df_combined.index.min()).total_seconds() / 3600
num_points = len(df_combined)
median_interval_sec = df_combined.index.to_series().diff().dropna().median().total_seconds()
print(f"\nMedian sampling interval = {median_interval_sec:.2f} seconds")
print(f"Total duration: {duration_hours:.2f} hours")
print(
    f"Median sampling interval: {median_interval_sec:.2f} seconds "
    f"(~{median_interval_sec/60:.1f} min, "
    f"{3600/median_interval_sec:.1f} samples/hour)"
)

# 3) MOVING AVERAGE WINDOWS based cuasal concept (Window: passt and current samples)

def window_points(minutes):
    return int(round(minutes * 60 / median_interval_sec))

WINDOWS = {
    "MA_30min": window_points(30),
    "MA_6h":    window_points(360),
    "MA_12h":   window_points(720),
}

# Apply smoothing
for name, w in WINDOWS.items():
    for col in ["Temperature", "Humidity", "Dewpoint"]:
        df_combined[f"{name}_{col}"] = (
            df_combined[col]
            .rolling(w, min_periods=1, center=False)
            .mean()
        )

# 4) PLOT ORIGINAL merged data#############################
print("\n2) Plotting original signals...")

plt.rcParams.update({
    "axes.linewidth": 1.2,
    "font.size": 11,
    "lines.linewidth": 2.2
})

fig, ax1 = plt.subplots(figsize=(16, 6))

# Left axis
ax1.plot(df_combined.index, df_combined["Temperature"], color="red", label="Temperature")
ax1.plot(df_combined.index, df_combined["Dewpoint"], color="green", linestyle=":", label="Dew Point")
ax1.set_ylabel("Temperature / Dew Point (°C)")
ax1.set_xlabel("Time")

# Right axis
ax2 = ax1.twinx()
ax2.plot(df_combined.index, df_combined["Humidity"], color="blue", linestyle="--", label="Humidity")
ax2.set_ylabel("Humidity (%)")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

ax1.grid(True, linestyle="--", alpha=0.35)
plt.title("Temperature, Humidity, & Dew Point — Original Signals")

plt.tight_layout()
plt.savefig(f"{OUTPUT_FIG_DIR}/Original_All_Signals.png", dpi=300)
plt.close()

#####Qualitative comparison###########################################################################
# 5) SUBPLOTS FOR SMOOTHED DATA #########  Compare fluctuation patterns in the original vs. smoothed signals across different moving-average window sizes

print("3) Generating subplot figures original vs smothed...")

signals = ["Temperature", "Humidity", "Dewpoint"]
units = ["°C", "%", "°C"]
ma_keys = list(WINDOWS.keys())
colors = {"MA_30min": "red", "MA_6h": "green", "MA_12h": "blue"}

for sig, unit in zip(signals, units):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f"{sig} — Original vs Smoothed")

    for i, ma_key in enumerate(ma_keys):
        ax = axes[i]
        ax.plot(df_combined.index, df_combined[sig], color="black", alpha=0.5, label="Original")
        ax.plot(df_combined.index, df_combined[f"{ma_key}_{sig}"], color=colors[ma_key], linewidth=2, label="Smoothed")

        duration_hr = round(WINDOWS[ma_key] * median_interval_sec / 3600, 2)
        ax.set_title(f"{ma_key} (~{duration_hr} hours)")
        ax.set_ylabel(unit)
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FIG_DIR}/{sig}_SUBPLOTS.png", dpi=300)
    plt.close()



# 6) Results Summary 
###Quantitative comparison

# 6.1) STATISTICAL COMPARISON using STD and Variance
###### Analyze fluctuation levels using standard deviation and variance for original vs. smoothed signals

print("\n4) Statistical comparison:")

stats_tables = {}
versions = ["Original", "MA_30min", "MA_6h", "MA_12h"]

for sig in signals:
    print(f"\n--- {sig} ---")
    rows = []

    for v in versions:
        col_name = sig if v == "Original" else f"{v}_{sig}"
        series = df_combined[col_name].dropna()

        stats = {
            "Version": v,
            "Std": round(series.std(), 3),
            "Variance": round(series.var(), 3),
        }

        rows.append(stats)

    df_stats = pd.DataFrame(rows)
    stats_tables[sig] = df_stats
    print(df_stats.to_string(index=False))

# 6.2) COMPARISON METRIC: ORIGINAL vs SMOOTHED uisng Root Mean Sequare Errro (RMSE) 

####### Lossing information evaulation (Original vs Smoothed)
print("\n7) COMPARISON METRICS using RMSE (Original vs Smoothed):")
comparison_metrics = {}

for sig in signals:
    print(f"\n### {sig} ###")
    
    orig = df_combined[sig].dropna()
    metrics_rows = []
    
    for v in ["MA_30min", "MA_6h", "MA_12h"]:
        smooth = df_combined[f"{v}_{sig}"].dropna()
        
        # Align them
        aligned = pd.concat([orig, smooth], axis=1).dropna()
        o = aligned.iloc[:,0]
        s = aligned.iloc[:,1]

        rmse = np.sqrt(np.mean((o - s)**2))
        corr = o.corr(s)
        var_retained = (np.var(s) / np.var(o)) * 100

        metrics_rows.append({
            "Version": v,
            "RMSE": round(rmse, 4),
        })

    df_metrics = pd.DataFrame(metrics_rows)
    comparison_metrics[sig] = df_metrics
    
    print(df_metrics.to_string(index=False))