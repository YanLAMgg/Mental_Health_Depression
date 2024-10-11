import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'Mental_health_Depression_disorder_Data.csv'
data = pd.read_csv(file_path)

# Handling mixed type columns by converting them to numeric
data['Schizophrenia (%)'] = pd.to_numeric(data['Schizophrenia (%)'], errors='coerce')
data['Bipolar disorder (%)'] = pd.to_numeric(data['Bipolar disorder (%)'], errors='coerce')
data['Eating disorders (%)'] = pd.to_numeric(data['Eating disorders (%)'], errors='coerce')
data['Anxiety disorders (%)'] = pd.to_numeric(data['Anxiety disorders (%)'], errors='coerce')
data['Drug use disorders (%)'] = pd.to_numeric(data['Drug use disorders (%)'], errors='coerce')
data['Depression (%)'] = pd.to_numeric(data['Depression (%)'], errors='coerce')
data['Alcohol use disorders (%)'] = pd.to_numeric(data['Alcohol use disorders (%)'], errors='coerce')

# Dropping rows with NaN values
data = data.dropna()

# Feature and target selection
X = data[['Year']]
y = data['Depression (%)']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creating and training the KNN regressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Predicting on test set
y_pred = knn.predict(X_test_scaled)

# Sorting the values for better visualization
X_test_flat = X_test.values.flatten()
sorted_indices = np.argsort(X_test_flat)
X_test_sorted = X_test_flat[sorted_indices]
y_test_sorted = y_test.values[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

# Plotting the results with sorted values
plt.figure(figsize=(14, 8))
plt.plot(X_test_sorted, y_test_sorted, 'bo', label='Actual Depression (%)')
plt.plot(X_test_sorted, y_pred_sorted, 'r-', label='Predicted Depression (%)', linewidth=2, alpha=0.7)
plt.title('KNN Regression: Year vs Depression (%)')
plt.xlabel('Year')
plt.ylabel('Depression (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
