import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import joblib

# Load the dataset
df = pd.read_csv('Sleep_Efficiency.csv', index_col=0)

# Explore the dataset
print(df.head())
print(df.info())
print(df.describe())

# Identify columns for clustering
feature_columns = [
    'Age', 'Sleep duration', 'Sleep efficiency',
    'REM sleep percentage', 'Deep sleep percentage',
    'Light sleep percentage', 'Awakenings',
    'Caffeine consumption', 'Alcohol consumption',
    'Exercise frequency'
]

# Check for missing values
print(df[feature_columns].isnull().sum())

# Impute missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
df[feature_columns] = imputer.fit_transform(df[feature_columns])

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_columns])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
print(df[['Cluster'] + feature_columns].groupby('Cluster').mean())

# Visualize clusters
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(10, 7))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='viridis')
plt.title('KMeans Clustering of Sleep Data')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()

# Save the scaler and model
joblib.dump(scaler, 'model/scaler.joblib')
joblib.dump(kmeans, 'model/kmeans_model.joblib')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import joblib

# Load the dataset
df = pd.read_csv('Sleep_Efficiency.csv', index_col=0)

# Explore the dataset
print(df.head())
print(df.info())
print(df.describe())

# Identify columns for clustering
feature_columns = [
    'Age', 'Sleep duration', 'Sleep efficiency',
    'REM sleep percentage', 'Deep sleep percentage',
    'Light sleep percentage', 'Awakenings',
    'Caffeine consumption', 'Alcohol consumption',
    'Exercise frequency'
]

# Check for missing values
print(df[feature_columns].isnull().sum())

# Impute missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
df[feature_columns] = imputer.fit_transform(df[feature_columns])

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_columns])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
print(df[['Cluster'] + feature_columns].groupby('Cluster').mean())

# Visualize clusters
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(10, 7))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='viridis')
plt.title('KMeans Clustering of Sleep Data')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()

# Save the scaler and model
joblib.dump(scaler, 'model/scaler.joblib')
joblib.dump(kmeans, 'model/kmeans_model.joblib')