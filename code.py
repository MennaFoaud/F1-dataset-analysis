# F1 Complete Data Visualization & Analysis Project (Google Colab Ready)

# Install required packages (Uncomment when running for the first time)
# !pip install plotly seaborn scikit-learn

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from google.colab import files
import io

# Upload all CSV files you saw in your folder
uploaded = files.upload()

# Load all relevant files
circuits = pd.read_csv(io.BytesIO(uploaded['circuits.csv']))
constructors = pd.read_csv(io.BytesIO(uploaded['constructors.csv']))
constructor_results = pd.read_csv(io.BytesIO(uploaded['constructor_results.csv']))
constructor_standings = pd.read_csv(io.BytesIO(uploaded['constructor_standings.csv']))
drivers = pd.read_csv(io.BytesIO(uploaded['drivers.csv']))
driver_standings = pd.read_csv(io.BytesIO(uploaded['driver_standings.csv']))
lap_times = pd.read_csv(io.BytesIO(uploaded['lap_times.csv']))
pit_stops = pd.read_csv(io.BytesIO(uploaded['pit_stops.csv']))
qualifying = pd.read_csv(io.BytesIO(uploaded['qualifying.csv']))
races = pd.read_csv(io.BytesIO(uploaded['races.csv']))
results = pd.read_csv(io.BytesIO(uploaded['results.csv']))
seasons = pd.read_csv(io.BytesIO(uploaded['seasons.csv']))
sprint_results = pd.read_csv(io.BytesIO(uploaded['sprint_results.csv']))
status = pd.read_csv(io.BytesIO(uploaded['status.csv']))

# --- ANALYSIS SECTION ---

# 1. Number of races per season
race_counts = races.groupby('year').count()['raceId']
race_counts.plot(figsize=(12,6), title='Number of Races per Season')
plt.xlabel('Season')
plt.ylabel('Number of Races')
plt.grid(True)
plt.show()

# 2. Top 10 winning drivers (by wins)
top_wins = results[results['positionOrder'] == 1].groupby('driverId').size().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 5))
top_wins.plot(kind='bar', color='darkred')
plt.title('Top 10 Drivers with Most Wins')
plt.ylabel('Wins')
plt.xlabel('Driver ID')
plt.show()

# 3. Pit stop durations distribution
pit_stops = pit_stops[pit_stops['milliseconds'].notna()]
sns.histplot(pit_stops['milliseconds'], bins=50, kde=True)
plt.title('Distribution of Pit Stop Durations')
plt.xlabel('Milliseconds')
plt.show()

# 4. Fastest average lap times by circuit
lap_times = lap_times.merge(races[['raceId', 'circuitId']], on='raceId')
lap_times = lap_times.merge(circuits[['circuitId', 'name']], on='circuitId')
lap_summary = lap_times.groupby('name')['milliseconds'].mean().sort_values().head(10)
fig = px.bar(lap_summary, title='Top 10 Fastest Circuits by Average Lap Time')
fig.show()

# 5. Driver Standings Trends (Optional: select a driverId)
example_driver = driver_standings[driver_standings['driverId'] == 1]
plt.plot(example_driver['raceId'], example_driver['points'])
plt.title('Example Driver Standings over Time')
plt.xlabel('Race ID')
plt.ylabel('Points')
plt.show()

# 6. Predictive Modeling for Lap Times
lap_times = lap_times.dropna(subset=['milliseconds', 'circuitId'])
X = lap_times[['circuitId']]
y = lap_times['milliseconds']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Lap Time Prediction RMSE:", rmse)

# 7. Constructor Standings Analysis
top_teams = constructor_standings.groupby('constructorId')['points'].sum().sort_values(ascending=False).head(10)
top_team_names = constructors[constructors['constructorId'].isin(top_teams.index)]
top_teams.index = top_team_names.set_index('constructorId').loc[top_teams.index]['name']
fig = px.bar(top_teams, title='Top 10 Teams by Total Points in Standings')
fig.show()

# 8. Qualifying performance distribution
qualifying['q1'] = pd.to_numeric(qualifying['q1'], errors='coerce')
sns.histplot(qualifying['q1'].dropna(), kde=True, bins=40)
plt.title('Qualifying Q1 Time Distribution')
plt.xlabel('Time (s)')
plt.show()
# Merge driver names into the top winners chart
results_with_names = results.merge(drivers[['driverId', 'forename', 'surname']], on='driverId')
results_with_names['driver_name'] = results_with_names['forename'] + ' ' + results_with_names['surname']

# Top 10 winning drivers (with names)
top_wins_named = results_with_names[results_with_names['positionOrder'] == 1].groupby('driver_name').size().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 5))
top_wins_named.plot(kind='bar', color='darkblue')
plt.title('Top 10 Winning Drivers by Name')
plt.ylabel('Wins')
plt.xlabel('Driver Name')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Merge constructor names into constructor standings analysis
top_teams_named = constructor_standings.groupby('constructorId')['points'].sum().sort_values(ascending=False).head(10)
constructor_names = constructors.set_index('constructorId').loc[top_teams_named.index]['name']
top_teams_named.index = constructor_names

# Plot with team names
fig = px.bar(top_teams_named, title='Top 10 Constructors by Total Points', labels={'value': 'Points', 'index': 'Constructor'})
fig.show()
# Merge driver names into driver standings trends
example_driver_named = driver_standings[driver_standings['driverId'] == 1].merge(drivers[['driverId', 'forename', 'surname']], on='driverId')