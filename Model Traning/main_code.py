import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('Lungs Cancer Detection/survey.csv')

# Convert categorical columns to numerical
df['GENDER'] = df['GENDER'].replace(['M', 'F'], [0, 1])
df['LUNG_CANCER'] = df['LUNG_CANCER'].replace(['YES', 'NO'], [1, 0])

# Split dataset into features and labels
X = df.drop(['LUNG_CANCER'], axis=1)
y = df['LUNG_CANCER']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Save the scaler
joblib.dump(sc, 'scaler.pkl')

# Build Neural Network model
classifier = Sequential()
classifier.add(Dense(units=12, activation='relu', input_dim=X_train.shape[1]))
classifier.add(Dense(units=8, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))  # Output probability

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, y_train, batch_size=10, epochs=150, verbose=0)

# Save the model
classifier.save('lung_cancer_model.h5')
print("Model saved successfully!")

# Evaluate model
score, acc = classifier.evaluate(X_test, y_test, batch_size=10)
print(f'Test Score: {score:.4f}, Test Accuracy: {acc:.4f}')

# Predict probabilities
y_pred_prob = classifier.predict(X_test) * 100  # Convert to percentage

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_pred_prob):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Confusion Matrix
y_pred_binary = (y_pred_prob > 50).astype(int)  # Convert probability to binary (50% threshold)
cm = confusion_matrix(y_test, y_pred_binary)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred_binary))

# Clustering to analyze lung cancer cases
df_cancer = df[df['LUNG_CANCER'] == 1].drop('LUNG_CANCER', axis=1)
df_cancer_scaled = sc.fit_transform(df_cancer)

# Determine optimal number of clusters (Elbow Method)
distortions = []
for i in range(1, 10):
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    km.fit(df_cancer_scaled)
    distortions.append(km.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 10), distortions, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.title('Elbow Method for Optimal Clusters')
plt.show()

# Perform clustering
kmeans_model = KMeans(n_clusters=5, random_state=10).fit(df_cancer_scaled)
df_cancer['Cluster'] = kmeans_model.labels_

# Visualizing clusters
plt.figure(figsize=(10, 6))
sns.countplot(x=df_cancer['Cluster'], palette='coolwarm')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('Distribution of Clusters among Lung Cancer Cases')
plt.show()

print(df_cancer.groupby('Cluster').mean().T)