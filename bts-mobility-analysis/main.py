"""
Sections:
- Weekly averages using Dask
- Trip distances grouped by week
- Filtering trips >10M and plotting
- Parallel processing comparison with Dask
- Linear Regression to simulate frequency model
- Visualization of actual vs predicted trips
"""

# === Import Required Libraries === #
import dask.dataframe as dd
import matplotlib.pyplot as plt
from dask.distributed import Client, LocalCluster
import time
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# === Question 1a: Weekly Average Staying at Home === #
df = dd.read_csv("Trips_by_Distance.csv", 
                 dtype={
                     'County Name': 'object',
                     'State Postal Code': 'object',
                     'Population Staying at Home': 'float64',
                     'Number of Trips 10-25': 'float64',
                     'Number of Trips 50-100': 'float64'
                 }, assume_missing=True)

df.columns = df.columns.str.strip()
weekly_avg = df.groupby('Week')['Population Staying at Home'].mean().compute()

plt.figure(figsize=(10, 5))
plt.bar(weekly_avg.index, weekly_avg.values, color='skyblue')
plt.xlabel('Week')
plt.ylabel('Average People Staying at Home')
plt.title('Weekly Average of People Staying at Home')
plt.tight_layout()
plt.savefig("weekly_avg_staying_home.png")
plt.show()

# === Load Full Distance Data & Analyze by Range === #
df_full = dd.read_csv("Trips_Full Data (2).csv", assume_missing=True)
df_full.columns = df_full.columns.str.strip()
df_full['Week'] = dd.to_datetime(df_full['Date'], errors='coerce').dt.isocalendar().week

distance_cols = ['Trips 1-25 Miles', 'Trips 25-100 Miles', 'Trips 100+ Miles']
weekly_distances = df_full.groupby('Week')[distance_cols].mean().compute()
weekly_distances.columns = ['1-25 Miles', '25-100 Miles', '100+ Miles']

weekly_distances.plot(kind='bar', figsize=(14, 6), width=0.8)
plt.xlabel("Week")
plt.ylabel("Average Number of Trips")
plt.title("Average Trips per Distance Range by Week")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("average_trips_by_distance_range.png")
plt.show()

# === Question 1b: Trips Above 10 Million === #
df = dd.read_csv("Trips_by_Distance.csv", assume_missing=True)
df.columns = df.columns.str.strip()

df_10_25 = df[df['Number of Trips 10-25'] > 10000000].compute()
df_50_100 = df[df['Number of Trips 50-100'] > 10000000].compute()

plt.figure(figsize=(12, 5))
plt.scatter(df_10_25['Date'], df_10_25['Number of Trips 10-25'], color='orange')
plt.xlabel('Date')
plt.ylabel('Number of Trips 10–25')
plt.title('Dates with >10M People Taking 10–25 Trips')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("scatter_10_25_trips.png")
plt.show()

plt.figure(figsize=(12, 5))
plt.scatter(df_50_100['Date'], df_50_100['Number of Trips 50-100'], color='green')
plt.xlabel('Date')
plt.ylabel('Number of Trips 50–100')
plt.title('Dates with >10M People Taking 50–100 Trips')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("scatter_50_100_trips.png")
plt.show()

# === Question 1c: Parallel Processing Performance === #
def run_with_workers(n_workers):
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=False)
    client = Client(cluster)

    df = dd.read_csv("Trips_by_Distance.csv",
                     dtype={
                         'County Name': 'object',
                         'State Postal Code': 'object',
                         'Population Staying at Home': 'float64',
                         'Number of Trips 10-25': 'float64',
                         'Number of Trips 50-100': 'float64'
                     }, assume_missing=True)
    df.columns = df.columns.str.strip()

    start = time.time()
    df.groupby('County Name')['Population Staying at Home'].mean().compute()
    end = time.time()

    client.close()
    cluster.close()
    return end - start

print("Running performance benchmark...")
print(f"⏱️ Time with 2 workers: {run_with_workers(2):.2f} seconds")
print(f"⏱️ Time with 4 workers: {run_with_workers(4):.2f} seconds")

# === Question 1d: Predictive Modeling with Regression === #
df_model = pd.read_csv("Trips_Full Data (2).csv")
df_model.columns = df_model.columns.str.strip()
df_model = df_model.dropna(subset=['Trips 1-25 Miles', 'Trips 25-100 Miles', 'Trips 100+ Miles', 'Trips'])

# Features and Target
X = df_model[['Trips 1-25 Miles', 'Trips 25-100 Miles', 'Trips 100+ Miles']]
y = df_model['Trips']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Model Summary
print("🎯 Coefficients:")
for name, coef in zip(X.columns, model.coef_):
    print(f"{name}: {coef:.2f}")
print(f"📊 R-squared: {model.score(X_test, y_test):.2f}")

# === Question 1e: Weekly Visualization of Distance Participation === #
df = pd.read_csv("Trips_Full Data (2).csv")
df.columns = df.columns.str.strip()
df = df.dropna(subset=['Trips 1-25 Miles', 'Trips 25-100 Miles', 'Trips 100+ Miles'])
df['Week'] = pd.to_datetime(df['Date'], errors='coerce').dt.isocalendar().week

weekly_travelers = df.groupby('Week')[['Trips 1-25 Miles', 'Trips 25-100 Miles', 'Trips 100+ Miles']].mean()
weekly_travelers.plot(kind='bar', figsize=(16, 6))
plt.title('Weekly Participation by Distance')
plt.xlabel('Week Number')
plt.ylabel('Average Travelers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("weekly_travelers_by_distance.png")
plt.show()

# === Extra: Model Testing - Actual vs Predicted Plot === #
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5, color='purple', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Actual Trips")
plt.ylabel("Predicted Trips")
plt.title("Actual vs Predicted Number of Trips")
plt.grid(True)
plt.tight_layout()
plt.savefig("model_test_actual_vs_predicted.png")
plt.show()
