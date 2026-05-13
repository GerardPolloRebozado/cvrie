import pandas as pd
import re
import csv
import joblib
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

data = []

with open('Student_Dataset.csv', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = list(csv.reader([line]))[0]

        if len(parts) == 2:
            data.append(
                {'ID': parts[0], 'Color': '0x000000', 'Testimony': parts[1]})
        elif len(parts) >= 3:
            data.append(
                {'ID': parts[0], 'Color': parts[1], 'Testimony': parts[2]})

df = pd.DataFrame(data)


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

kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
kmeans.fit(X)

terms = tfidf.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

cluster_keywords = {}

for i in range(kmeans.n_clusters):
    top_words = [terms[ind] for ind in order_centroids[i, :8]]
    cluster_keywords[i] = top_words

joblib.dump(tfidf, "vectorizer.pkl")
joblib.dump(kmeans, "kmeans_model.pkl")

with open("cluster_keywords.json", "w") as f:
    json.dump(cluster_keywords, f)

print("Model and keywords saved.")
