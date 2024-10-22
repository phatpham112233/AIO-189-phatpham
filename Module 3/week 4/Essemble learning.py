import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


dataset_path = 'C:\Users\Admin\AIO-189-phatpham-1\Module 3\week 4\Housing.csv'
df = pd.read_csv(dataset_path)


print(df.head())


categorical_cols = df.select_dtypes(include=['object']).columns.to_list()


ordinal_encoder = OrdinalEncoder()
encoded_categorical_cols = ordinal_encoder.fit_transform(df[categorical_cols])


encoded_categorical_df = pd.DataFrame(encoded_categorical_cols, columns=categorical_cols)


numerical_df = df.drop(categorical_cols, axis=1)
encoded_df = pd.concat([numerical_df, encoded_categorical_df], axis=1)

print(encoded_df.head()) 


scaler = StandardScaler()
dataset_arr = scaler.fit_transform(encoded_df)


X, y = dataset_arr[:, 1:], dataset_arr[:, 0] 

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1, shuffle=True)




dt_regressor = DecisionTreeRegressor(random_state=1)
dt_regressor.fit(X_train, y_train)


rf_regressor = RandomForestRegressor(random_state=1)
rf_regressor.fit(X_train, y_train)




y_pred_dt = dt_regressor.predict(X_val)
mae_dt = mean_absolute_error(y_val, y_pred_dt)
mse_dt = mean_squared_error(y_val, y_pred_dt)


y_pred_rf = rf_regressor.predict(X_val)
mae_rf = mean_absolute_error(y_val, y_pred_rf)
mse_rf = mean_squared_error(y_val, y_pred_rf)


print("Decision Tree:")
print(f"MAE: {mae_dt}, MSE: {mse_dt}")
print("\nRandom Forest:")
print(f"MAE: {mae_rf}, MSE: {mse_rf}")

