

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Funktion um Word Cloud eines beliebigen Korpus zu generieren
def generate_wordcloud_from_csv(csv_path, width=800, height=400):
    # Load den Korpus
    df = pd.read_csv(csv_path)

    # Anzahl der Fräquenz jedes Wort 
    word_freq = df.sum().to_dict()

    # Cloud generieren
    wc = WordCloud(width=width, height=height, background_color="white")
    wc.generate_from_frequencies(word_freq)

    return wc




# Word Cloud für Thema LDA
def generate_topic_wordcloud(lda_model, topic_id, vocab, width=1200, height=920, topn=30):
    topic = lda_model.components_[topic_id]
    top_indices = topic.argsort()[::-1][:topn]
    topic_terms = {vocab[i]: topic[i] for i in top_indices}
    
    wc = WordCloud(width=width, height=height, background_color="white").generate_from_frequencies(topic_terms)
    return wc


# Bar Chart für Thema LDA
def generate_topic_bar_chart(lda_model, topic_id, vocab, topn=10):
    topic = lda_model.components_[topic_id]
    top_indices = topic.argsort()[::-1][:topn]
    words = [vocab[i] for i in top_indices]
    weights = [topic[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(words[::-1], weights[::-1])
    ax.set_xlabel("Pes")
    ax.set_title(f"Tema {topic_id + 1}")
    return fig



# SpaCy
def plot_word_vectors(nlp, words):
    word_vectors = [nlp(w)[0].vector for w in words if nlp(w)[0].has_vector]
    labels = [w for w in words if nlp(w)[0].has_vector]

    if not word_vectors:
        return None

    pca = PCA(n_components=2)
    coords = pca.fit_transform(word_vectors)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[:, 0], coords[:, 1])

    for i, label in enumerate(labels):
        ax.annotate(label, (coords[i, 0], coords[i, 1]))

    ax.set_title("spaCy – PCA de vectors semàntics")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True)

    return fig
      