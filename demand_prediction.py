#chaitanyavamsimanukonda.github.io

'''Demand Prediction with Parameter Tuning, Programming for Business, April 2023'''

''' Importing necessary libraries like pandas and numpy for numerical computing and 
data analysis,
modules like model_selection, ensemble, metrics
for splitting data into train and test data,
fit a random forest regression model to the training data 
and evaluate the performance of the model respectively. '''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Loading dataset into a pandas dataframe
df = pd.read_csv("grocery_store_data.csv")

# Converting the 'week' column to a Unix timestamp
df['week'] = pd.to_datetime(df['week'], format='%d/%m/%y')
df['week'] = df['week'].astype(np.int64) // 10**9

# Removing the duplicates
df.drop_duplicates(inplace=True)

# Removing the missing data
df.dropna(inplace=True)

# Defining the features and target variable
X = df.drop("units_sold", axis=1)
y = df["units_sold"]

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42)

# Defining a range of values for the parameters to search
n_estimators_range = [10, 100, 150]
max_depth_range = [10, 30, 50, None]
max_leaf_nodes_range = [10, 30, 50, None]

# Initializing the best parameters and the best scores
best_params = None
best_r2 = float('-inf')
best_rmse = float('inf')

# Looping through the parameter grid
for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        for max_leaf_nodes in max_leaf_nodes_range:
            # Instantiation of the Random Forest Regressor with the current parameters
            rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
            max_leaf_nodes=max_leaf_nodes, random_state=42)

            # Training the model
            rf.fit(X_train, y_train)

            # Prediction and evaluation of the model with R2 and RMSE
            y_pred = rf.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            # Updating the best parameters and the best scores if both conditions are 
            met
            if r2 > best_r2 and rmse < best_rmse:
                best_params = {'n_estimators': n_estimators, 'max_depth': max_depth,
                'max_leaf_nodes': max_leaf_nodes}
                best_r2 = r2
                best_rmse = rmse

# Print the best parameters and the best scores
print(f"Best parameters: {best_params}")
print(f"Best R2 score: {best_r2:.3f}")
print(f"Best RMSE: {best_rmse:.3f}")
