import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Linear Algebra in NLP",
    page_icon="📐",
    layout="wide"
)

sns.set_theme(style="whitegrid")

EMBEDDINGS = {
    "king": np.array([0.90, 0.80, 0.10, 0.05, 0.10, 0.05]),
    "queen": np.array([0.86, 0.88, 0.10, 0.05, 0.10, 0.05]),
    "man": np.array([0.70, 0.20, 0.05, 0.02, 0.05, 0.02]),
    "woman": np.array([0.66, 0.28, 0.05, 0.02, 0.05, 0.02]),
    "prince": np.array([0.82, 0.70, 0.08, 0.05, 0.10, 0.05]),
    "princess": np.array([0.78, 0.78, 0.08, 0.05, 0.10, 0.05]),

    "dog": np.array([0.05, 0.08, 0.95, 0.75, 0.10, 0.05]),
    "cat": np.array([0.05, 0.10, 0.90, 0.80, 0.10, 0.04]),
    "puppy": np.array([0.04, 0.07, 0.98, 0.70, 0.08, 0.05]),
    "kitten": np.array([0.04, 0.11, 0.92, 0.78, 0.08, 0.04]),
    "pet": np.array([0.05, 0.09, 0.88, 0.82, 0.09, 0.04]),

    "basketball": np.array([0.10, 0.05, 0.05, 0.05, 0.95, 0.80]),
    "team": np.array([0.12, 0.05, 0.04, 0.05, 0.90, 0.72]),
    "sport": np.array([0.10, 0.05, 0.04, 0.05, 0.92, 0.78]),
    "coach": np.array([0.14, 0.04, 0.04, 0.05, 0.84, 0.68]),

    "doctor": np.array([0.08, 0.05, 0.06, 0.05, 0.05, 0.95]),
    "nurse": np.array([0.07, 0.06, 0.05, 0.05, 0.05, 0.90]),
    "hospital": np.array([0.06, 0.05, 0.05, 0.05, 0.06, 0.88]),
    "patient": np.array([0.07, 0.05, 0.05, 0.05, 0.05, 0.86]),
}

WORDS = sorted(EMBEDDINGS.keys())


def cosine_similarity(u, v):
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom == 0:
        return 0.0
    return float(np.dot(u, v) / denom)


def nearest_words(query_word, top_n=5):
    q = EMBEDDINGS[query_word]
    rows = []
    for word, vec in EMBEDDINGS.items():
        if word != query_word:
            rows.append({
                "word": word,
                "cosine_similarity": cosine_similarity(q, vec)
            })
    return (
        pd.DataFrame(rows)
        .sort_values("cosine_similarity", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def nearest_to_vector(vector, exclude_words=None, top_n=5):
    exclude_words = set(exclude_words or [])
    rows = []
    for word, vec in EMBEDDINGS.items():
        if word not in exclude_words:
            rows.append({
                "word": word,
                "cosine_similarity": cosine_similarity(vector, vec)
            })
    return (
        pd.DataFrame(rows)
        .sort_values("cosine_similarity", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


st.sidebar.title("About this app")
st.sidebar.write(
    "This app demonstrates word vectors, cosine similarity, vector analogies, "
    "and dimensionality reduction."
)
st.sidebar.info(
    "The embeddings are small demo vectors, not pretrained production embeddings."
)

st.title("Linear Algebra in Modern NLP")
st.write(
    "Explore how words can be represented as vectors and how geometric relationships "
    "between vectors can encode semantic relationships between words."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Word Similarity",
    "Analogy Explorer",
    "Embedding Visualization",
    "Theory Notes"
])

with tab1:
    st.header("Word Similarity Explorer")
    st.write(
        "Choose a word and compute its nearest neighbors using cosine similarity."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        query_word = st.selectbox(
            "Choose a word",
            WORDS,
            index=WORDS.index("dog")
        )
    with col2:
        top_n = st.slider("Number of nearest words", 3, 10, 5)

    results = nearest_words(query_word, top_n=top_n)
    st.subheader(f"Nearest words to '{query_word}'")
    st.dataframe(results, use_container_width=True)

    st.markdown("### Formula")
    st.latex(r"\cos(\theta)=\frac{u\cdot v}{\|u\|\|v\|}")

with tab2:
    st.header("Analogy Explorer")
    st.write(
        "This tab demonstrates vector arithmetic in embedding spaces. "
        "The classic example is: king - man + woman ≈ queen."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        word_a = st.selectbox("Positive word 1", WORDS, index=WORDS.index("king"))
    with c2:
        word_b = st.selectbox("Subtract word", WORDS, index=WORDS.index("man"))
    with c3:
        word_c = st.selectbox("Positive word 2", WORDS, index=WORDS.index("woman"))

    analogy_vector = EMBEDDINGS[word_a] - EMBEDDINGS[word_b] + EMBEDDINGS[word_c]
    analogy_results = nearest_to_vector(
        analogy_vector,
        exclude_words={word_a, word_b, word_c},
        top_n=5
    )

    st.latex(rf"\text{{{word_a}}} - \text{{{word_b}}} + \text{{{word_c}}} \approx ?")
    st.subheader("Nearest words to the resulting vector")
    st.dataframe(analogy_results, use_container_width=True)

with tab3:
    st.header("2D Embedding Visualization")
    st.write(
        "The demo embeddings live in 6 dimensions. PCA reduces them to 2 dimensions "
        "so we can visualize their geometric relationships."
    )

    selected_words = st.multiselect(
        "Choose words to visualize",
        WORDS,
        default=["king", "queen", "man", "woman", "dog", "cat", "puppy", "basketball", "team"]
    )

    if len(selected_words) < 2:
        st.warning("Please choose at least two words.")
    else:
        X = np.vstack([EMBEDDINGS[w] for w in selected_words])
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

        plot_df = pd.DataFrame({
            "word": selected_words,
            "PC1": X_2d[:, 0],
            "PC2": X_2d[:, 1]
        })

        fig, ax = plt.subplots(figsize=(10, 7))

        sns.scatterplot(
            data=plot_df,
            x="PC1",
            y="PC2",
            s=120,
            ax=ax
        )

        offsets = [
            (8, 8), (8, -14), (-45, 8), (-45, -14),
            (12, 18), (12, -24), (-60, 18), (-60, -24),
            (20, 0), (-70, 0), (0, 24), (0, -30)
        ]

        for i, row in plot_df.iterrows():
            dx, dy = offsets[i % len(offsets)]
            ax.annotate(
                row["word"],
                (row["PC1"], row["PC2"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75),
                arrowprops=dict(arrowstyle="-", lw=0.6, alpha=0.6)
            )

        ax.set_title("PCA Projection of Word Embeddings", fontsize=15, pad=15)
        ax.set_xlabel("First principal component")
        ax.set_ylabel("Second principal component")

        x_margin = (plot_df["PC1"].max() - plot_df["PC1"].min()) * 0.25
        y_margin = (plot_df["PC2"].max() - plot_df["PC2"].min()) * 0.25

        ax.set_xlim(plot_df["PC1"].min() - x_margin, plot_df["PC1"].max() + x_margin)
        ax.set_ylim(plot_df["PC2"].min() - y_margin, plot_df["PC2"].max() + y_margin)

        st.pyplot(fig)

        st.write("Explained variance ratio:")
        st.dataframe(pd.DataFrame({
            "component": ["PC1", "PC2"],
            "explained_variance_ratio": pca.explained_variance_ratio_
        }), use_container_width=True)

with tab4:
    st.header("Theory Notes")

    st.subheader("Words as Vectors")
    st.write(
        "In NLP, words can be represented as vectors in a high-dimensional vector space. "
        "Once words are vectors, linear algebra tools such as dot products, norms, "
        "cosine similarity, and matrix decompositions become available."
    )

    st.subheader("Cosine Similarity")
    st.latex(r"\cos(\theta)=\frac{u\cdot v}{\|u\|\|v\|}")
    st.write(
        "Cosine similarity measures whether two vectors point in similar directions. "
        "In embedding spaces, similar directions often correspond to similar linguistic contexts and semantic patterns."
    )

    st.subheader("Dimensionality Reduction")
    st.write(
        "High-dimensional embeddings are difficult to visualize directly. PCA and SVD "
        "can project vectors into lower-dimensional spaces while preserving important structure."
    )
    st.latex(r"A \approx U_k \Sigma_k V_k^T")

    st.subheader("Important Note")
    st.write(
        "This app uses intentionally simple demo vectors that are pre-configured. In a full NLP system, embeddings "
        "would be learned from a large corpus using a method such as Word2Vec, GloVe, or a transformer model."
    )