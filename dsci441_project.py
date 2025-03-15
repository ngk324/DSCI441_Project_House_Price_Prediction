# -*- coding: utf-8 -*-
"""DSCI441_Project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rAa0BsEsSOjCAL7UbszXqy0m4iMP3nYS
"""

!unzip original_extracted_df.csv.zip

import pandas as pd
import numpy as np
import tensorflow as tf
import statsmodels.api as sm

#from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from IPython.display import clear_output, display, HTML

clear_output()

data = pd.read_csv('original_extracted_df.csv')
data.head()

print(len(data))
data = data.drop('ConvertedLot', axis=1)
data = data.drop('City', axis=1)
data = data.drop('State', axis=1)
data = data.drop('Street', axis=1)
data = data.drop('RentEstimate', axis=1)
#data = data.drop('MarketEstimate', axis=1)
data.dropna(inplace=True)
print(len(data))

mape = mean_absolute_percentage_error(data['MarketEstimate'],data['Price'])
print(mape)

data = pd.DataFrame(data)

# Conversion to sqft
ACRE_TO_SQFT = 43560

for index, row in data.iterrows():
    if row['LotUnit'] == 'acres':
        data.at[index, 'LotArea'] *= ACRE_TO_SQFT

print(data)
data = data.drop('LotUnit', axis=1)

data.head()

# features and target
X_ols = data.drop('Price', axis=1)
y_ols = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X_ols, y_ols, test_size=0.1, random_state=1)

# OLS regression
model = sm.OLS(y_train, X_train).fit()

# regression summary
#print(model.summary())

# predict
y_pred = model.predict(X_test)

# model evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
mape = mean_absolute_percentage_error(y_test, y_pred)
print("Mean Absolute Percentage Error:", mape)
RMSE = np.sqrt(np.mean((y_test - y_pred)**2))
print(RMSE)

print(y_pred[0:10])
print(y_test[0:10])

# Preprocess data
X = data.drop('Price', axis=1)
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build NN
model_nn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),  # Input layer
    Dense(32, activation='relu'),  # Hidden layer
    Dense(16, activation='relu'),  # Hidden layer
    Dense(1)  # Output layer (no activation for regression)
])

model_nn.compile(optimizer='adam', loss='mean_squared_error')

# Train model model
history = model_nn.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Predict
y_pred_nn = model_nn.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred_nn)
print(f"Mean Squared Error (MSE): {mse}")
mape = mean_absolute_percentage_error(y_test, y_pred_nn)
print("Mean Absolute Percentage Error:", mape)
#RMSE = np.sqrt(np.mean((y_test - y_pred_nn)**2))
#print(RMSE)

# features and target
X = data.drop('Price', axis=1)
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Random Forest Regressor
model_rf = RandomForestRegressor(random_state=1)
model_rf.fit(X_train, y_train)

# Predict
y_pred_rf = model_rf.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred_rf)
print(f"Mean Squared Error (MSE): {mse}")
mape = mean_absolute_percentage_error(y_test, y_pred_rf)
print("Mean Absolute Percentage Error:", mape)
RMSE = np.sqrt(np.mean((y_test - y_pred_rf)**2))
print(RMSE)

!kaggle datasets download -d ahmedshahriarsakib/usa-real-estate-dataset

!unzip usa-real-estate-dataset.zip

!pip install ydata-profiling
!pip install tensorflow

import pandas as pd
import numpy as np
import tensorflow as tf
import statsmodels.api as sm

from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from IPython.display import clear_output, display, HTML

clear_output()

import pandas as pd

data = pd.read_csv('realtor-data.zip.csv')
data.head()

profile = ProfileReport(data, title="Housing Prices")
profile

# Only contain sold house data and remove status feature because all the status' are sold
data = data[data['status'] ==  'sold']
data = data.drop('status', axis=1)
data = data.reset_index(drop=True)

# Remove sold date as all are in 2021 and 2022. Contains many NaN and not and 2 year span is not an important feature for housing
data = data.drop('prev_sold_date', axis=1)

# Remove NaN values
data.dropna(inplace=True)

# Make categorical variables of City and State numerical
data['city'] = pd.factorize(data['city'])[0]
data['state'] = pd.factorize(data['state'])[0]

print(len(data))
data.head()

# features and target
X_ols = data.drop('price', axis=1)
y_ols = data['price']

X_train, X_test, y_train, y_test = train_test_split(X_ols, y_ols, test_size=0.2, random_state=1)

# OLS regression
model = sm.OLS(y_train, X_train).fit()

# regression summary
#print(model.summary())

# predict
y_pred = model.predict(X_test)

# model evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
mape = mean_absolute_percentage_error(y_test, y_pred)
print("Mean Absolute Percentage Error:", mape)
RMSE = np.sqrt(np.mean((y_test - y_pred)**2))
print(RMSE)

# features and target
X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Random Forest Regressor
model_rf = RandomForestRegressor(random_state=1)
model_rf.fit(X_train, y_train)

# Predict
y_pred_rf = model_rf.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred_rf)
print(f"Mean Squared Error (MSE): {mse}")
mape = mean_absolute_percentage_error(y_test, y_pred_rf)
print("Mean Absolute Percentage Error:", mape)
RMSE = np.sqrt(np.mean((y_test - y_pred_rf)**2))
print(RMSE)

importances = model_rf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)

# Preprocess data
X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build NN
model_nn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),  # Input layer
    Dense(32, activation='relu'),  # Hidden layer
    Dense(16, activation='relu'),  # Hidden layer
    Dense(1)  # Output layer (no activation for regression)
])

# Compile and train model
model_nn.compile(optimizer='adam', loss='mean_squared_error')

history = model_nn.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Predict
y_pred_nn = model_nn.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred_nn)
print(f"Mean Squared Error (MSE): {mse}")
mape = mean_absolute_percentage_error(y_test, y_pred_nn)
print("Mean Absolute Percentage Error:", mape)
RMSE = np.sqrt(np.mean((y_test - y_pred_nn)**2))
print(RMSE)