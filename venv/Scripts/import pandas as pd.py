import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from pymongo import MongoClient
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Connect to the MongoDB client
client = MongoClient('mongodb+srv://CE:project1@mzuceproject.kyvn8sq.mongodb.net/')
db = client['MZU']  # Database name
collection = db['CE']  # Collection name

# Fetch the data from MongoDB
data = list(collection.find())

# Convert the fetched data to a DataFrame
df = pd.DataFrame(data)

# Print the columns and first few rows to debug
print("Columns in the DataFrame:", df.columns)
print("First few rows of the DataFrame:\n", df.head())

# Rename columns and convert 'ds' to datetime format
df.rename(columns={'DATE': 'ds', 'PRICE': 'y'}, inplace=True)
df['ds'] = pd.to_datetime(df['ds'])

# Set 'ds' as the index
df.set_index('ds', inplace=True)

# Exclude the time between 9 PM to 6 AM
df2 = df.between_time('06:00', '19:00')

# Print the first few rows after filtering
print("First few rows after filtering:\n", df2.head())

# Fit the ARIMA model on the dataset
order = (5, 1, 0)  # Example order; (p, d, q) should be chosen based on data
model = ARIMA(df2['y'], order=order)
model_fit = model.fit()

# Print the summary statistics
print(model_fit.summary())

# Create a dataframe with future dates for forecasting
future_dates = pd.date_range(start=df2.index[-1], periods=2*365*24, freq='H')
future_dates = future_dates[future_dates.indexer_between_time('07:00', '19:00')]

# Make the forecast
forecast = model_fit.get_forecast(steps=len(future_dates))
forecast_df = forecast.conf_int()
forecast_df['yhat'] = forecast.predicted_mean
forecast_df['ds'] = future_dates

# Display the forecast
print("Forecast tail:\n", forecast_df.tail())

# Plot the forecast
plt.figure(figsize=(10, 6))
plt.plot(df2.index, df2['y'], label='Observed')
plt.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast')
plt.fill_between(forecast_df['ds'], forecast_df['lower y'], forecast_df['upper y'], color='k', alpha=0.1)
plt.title('Forecast using ARIMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()