import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Customer_behaviour_Tourism.csv')

# Data cleaning: Remove missing values
df = df.dropna()

# Select features to be standardized
features_to_scale = ['Yearly_avg_view_on_travel_page', 'total_likes_on_outstation_checkin_given',
                     'Yearly_avg_comment_on_travel_page', 'total_likes_on_outofstation_checkin_received',
                     'Daily_Avg_mins_spend_on_traveling_page']

# Data standardization
scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Add sample comments for sentiment analysis
df['comment'] = ["Enjoying a beautiful day at the park!",
                 "Traffic was terrible this morning.",
                 "Just finished an amazing workout!",
                 "Had a productive day at work!",
                 "Excited for the weekend!"] * (len(df) // 5)

# Extract sentiment scores from comments
df['sentiment'] = df['comment'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Generate hypothetical timestamp column
df['timestamp'] = pd.date_range(start='1/1/2020', periods=len(df), freq='D')

# Extract time features
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Machine learning classification and clustering
X = df[features_to_scale]

# Create random labels for the example
df['label'] = np.random.randint(0, 2, df.shape[0])
y = df['label']

# Initialize classifiers
rf = RandomForestClassifier()
dt = DecisionTreeClassifier()
svm = SVC()

# Train classifiers
rf.fit(X, y)
dt.fit(X, y)
svm.fit(X, y)

# Predict and evaluate
y_pred_rf = rf.predict(X)
y_pred_dt = dt.predict(X)
y_pred_svm = svm.predict(X)

# Plot confusion matrices
cm_rf = confusion_matrix(y, y_pred_rf)
cm_dt = confusion_matrix(y, y_pred_dt)
cm_svm = confusion_matrix(y, y_pred_svm)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ConfusionMatrixDisplay(cm_rf).plot(ax=ax[0])
ax[0].title.set_text('Random Forest')
ConfusionMatrixDisplay(cm_dt).plot(ax=ax[1])
ax[1].title.set_text('Decision Tree')
ConfusionMatrixDisplay(cm_svm).plot(ax=ax[2])
ax[2].title.set_text('SVM')
plt.show()

# Display processed data using pandas
print(df.head())

# Example of mathematical formula applications

# Random Forest Classifier
# Assume there are 3 decision trees with predictions 0, 1, 1
T1 = 0
T2 = 1
T3 = 1
y_hat = (T1 + T2 + T3) / 3
print(f"Random Forest Prediction: {y_hat}")

# Decision Tree Classifier
# Assume feature a splits the dataset into T1 and T2 with entropies 0.5 and 0.3, and |T1| = 40, |T2| = 60
Entropy_T = 1.0
