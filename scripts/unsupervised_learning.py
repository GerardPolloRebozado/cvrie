import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

data = []
with open('Student_Dataset.csv', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        import csv
        parts = list(csv.reader([line]))[0]

        if len(parts) == 2:
            data.append(
                {'ID': parts[0], 'Color': '0x000000', 'Testimony': parts[1]})
        elif len(parts) >= 3:
            data.append(
                {'ID': parts[0], 'Color': parts[1], 'Testimony': parts[2]})

df = pd.DataFrame(data)
print(f"Loaded {len(df)} records.")
df.head()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


df['cleaned_testimony'] = df['Testimony'].apply(clean_text)

tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=1000,
    ngram_range=(1, 2)
)
X = tfidf.fit_transform(df['cleaned_testimony'])
print(f"TF-IDF Matrix shape: {X.shape}")

wcss = []
sil_scores = []
k_range = range(5, 25)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X, kmeans.labels_))

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_xlabel('Number of clusters (k)')
ax1.set_ylabel('WCSS (Inertia)', color='tab:blue')
ax1.plot(k_range, wcss, marker='o', color='tab:blue', label='WCSS')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Silhouette Score', color='tab:red')
ax2.plot(k_range, sil_scores, marker='x', color='tab:red', label='Silhouette')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('Elbow Method and Silhouette Score for Optimal k')
plt.show()

optimal_k = 12
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)

order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = tfidf.get_feature_names_out()

print("Top terms per cluster:")
for i in range(optimal_k):
    print(f"Cluster {i}: ", end='')
    for ind in order_centroids[i, :8]:
        print(f'{terms[ind]}, ', end='')
    print()

tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
X_embedded = tsne.fit_transform(X.toarray())

plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    X_embedded[:, 0], X_embedded[:, 1], c=df['cluster'], cmap='tab20', alpha=0.6)
plt.legend(*scatter.legend_elements(), title="Clusters",
           bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('t-SNE Visualization of Symptom Clusters')
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.show()
