import pandas as pd
from scipy.stats import linregress
import numpy as np
from pathlib import Path

# Load file (choose one)
path = Path("~/Downloads/Combined Stocks & ETFs.xlsx").expanduser()
df = pd.read_excel(path, sheet_name="stocks")

# Sort properly
df = df.sort_values(["Ticker", "Date"])

# (Optional) remove tickers with too little history
df = df.groupby("Ticker").filter(lambda g: len(g) >= 252)

def calculate_trend(group):
    # trend on log price for scale-invariant slope
    y = np.log(group["Close"].values)
    x = np.arange(len(y))
    slope, _, _, _, _ = linregress(x, y)
    return slope

def max_drawdown(group):
    cumulative_max = group["Close"].cummax()
    drawdown = (group["Close"] - cumulative_max) / cumulative_max
    return drawdown.min()

# Compute daily returns
df["Return"] = df.groupby("Ticker")["Close"].pct_change()

volatility = df.groupby("Ticker")["Return"].std()
mean_return = df.groupby("Ticker")["Return"].mean()
avg_volume = df.groupby("Ticker")["Volume"].mean()
trend = df.groupby("Ticker").apply(calculate_trend)
drawdown = df.groupby("Ticker").apply(max_drawdown)

features = pd.DataFrame({
    "MeanReturn": mean_return,
    "Volatility": volatility,
    "AvgVolume": avg_volume,
    "Trend": trend,
    "MaxDrawdown": drawdown
}).dropna()

features.reset_index(inplace=True)

# Log transform volume safely
features["LogAvgVolume"] = np.log1p(features["AvgVolume"])
features.drop(columns=["AvgVolume"], inplace=True)

# Remove extreme outliers (volatility top 1%)
features = features[features["Volatility"] < features["Volatility"].quantile(0.99)]

features.to_csv("stock_behavior_features.csv", index=False)