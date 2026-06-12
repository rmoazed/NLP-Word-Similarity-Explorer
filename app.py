import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Linear Algebra in NLP",
    page_icon="📐",
    layout="wide"
)

# ------------------------------------------------------------
# Demo embeddings
# ------------------------------------------------------------
# These are small, hand-built vectors designed to demonstrate the
# linear algebra ideas in the project. They are not trained embeddings.
# The app can later be upgraded to use pretrained Word2Vec/GloVe vectors.

EMBEDDINGS = {
    # royalty / gender-ish example
    "king": np.array([0.90, 0.80, 0.10, 0.05, 0.10, 0.05]),
    "queen": np.array([0.86, 0.88, 0.10, 0.05, 0.10, 0.05]),
    "man": np.array([0.70, 0.20, 0.05, 0.02, 0.05, 0.02]),
    "woman": np.array([0.66, 0.28, 0.05, 0.02, 0.05, 0.02]),
    "prince": np.array([0.82, 0.70, 0.08, 0.05, 0.10, 0.05]),
    "princess": np.array([0.78, 0.78, 0.08, 0.05, 0.10, 0.05]),

    # animal cluster
    "dog": np.array([0.05, 0.08, 0.95, 0.75, 0.10, 0.05]),
    "cat": np.array([0.05, 0.10, 0.90, 0.80, 0.10, 0.04]),
    "puppy": np.array([0.04, 0.07, 0.98, 0.70, 0.08, 0.05]),
    "kitten": np.array([0.04, 0.11, 0.92, 0.78, 0.08, 0.04]),
    "pet": np.array([0.05, 0.09, 0.88, 0.82, 0.09, 0.04]),

    # sports cluster
    "basketball": np.array([0.10, 0.05, 0.05, 0.05, 0.95, 0.80]),
    "team": np.array([0.12, 0.05, 0.04, 0.05, 0.90, 0.72]),
    "sport": np.array([0.10, 0.05, 0.04, 0.05, 0.92, 0.78]),
    "coach": np.array([0.14, 0.04, 0.04, 0.05, 0.84, 0.68]),

    # medicine cluster
    "doctor": np.array([0.08, 0.05, 0.06, 0.05, 0.05, 0.95]),
    "nurse": np.array([0.07, 0.06, 0.05, 0.05, 0.05, 0.90]),
    "hospital": np.array([0.06, 0.05, 0.05, 0.05, 0.06, 0.88]),
    "patient": np.array([0.07, 0.05, 0.05, 0.05, 0.05, 0.86]),
}

WORDS = sorted(EMBEDDINGS.keys())


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Return cosine similarity between two nonzero vectors."""
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom == 0:
        return 0.0
    return float(np.dot(u, v) / denom)


def nearest_words(query_word: str, top_n: int = 5) -> pd.DataFrame:
    """Find nearest words by cosine similarity."""
    q = EMBEDDINGS[query_word]
    rows = []
    for word, vec in EMBEDDINGS.items():
        if word == query_word:
            continue
        rows.append({
            "word": word,
            "cosine_similarity": cosine_similarity(q, vec)
        })
    df = pd.DataFrame(rows).sort_values("cosine_similarity", ascending=False)
    return df.head(top_n).reset_index(drop=True)


def nearest_to_vector(vector: np.ndarray, exclude_words=None, top_n: int = 5) -> pd.DataFrame:
    """Find nearest words to an arbitrary vector."""
    exclude_words = set(exclude_words or [])
    rows = []
    for word, vec in EMBEDDINGS.items():
        if word in exclude_words:
            continue
        rows.append({
            "word": word,
            "cosine_similarity": cosine_similarity(vector, vec)
        })
    df = pd.DataFrame(rows).sort_values("cosine_similarity", ascending=False)
    return df.head(top_n).reset_index(drop=True)


def embedding_dataframe() -> pd.DataFrame:
    """Return embeddings as a DataFrame."""
    return pd.DataFrame.from_dict(EMBEDDINGS, orient="index")


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("About this app")
st.sidebar.write(
    "This Streamlit app demonstrates how linear algebra appears in NLP: "
    "word vectors, cosine similarity, vector analogies, and dimensionality reduction."
)
st.sidebar.info(
    "The embeddings here are small demo vectors, not pretrained production embeddings. "
    "They are intentionally simple so the linear algebra is easy to see."
)

# ------------------------------------------------------------
# Main app
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Tab 1: Word Similarity
# ------------------------------------------------------------
with tab1:
    st.header("Word Similarity Explorer")
    st.write(
        "Choose a word and compute its nearest neighbors using cosine similarity. "
        "Cosine similarity measures the angle between two vectors."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        query_word = st.selectbox("Choose a word", WORDS, index=WORDS.index("dog") if "dog" in WORDS else 0)
    with col2:
        top_n = st.slider("Number of nearest words", 3, 10, 5)

    results = nearest_words(query_word, top_n=top_n)
    st.subheader(f"Nearest words to '{query_word}'")
    st.dataframe(results, use_container_width=True)

    st.markdown("### Formula")
    st.latex(r"\cos(\theta)=\frac{u\cdot v}{\|u\|\|v\|}")

# ------------------------------------------------------------
# Tab 2: Analogy Explorer
# ------------------------------------------------------------
with tab2:
    st.header("Analogy Explorer")
    st.write(
        "This tab demonstrates vector arithmetic in embedding spaces. "
        "The classic example is: king - man + woman ≈ queen."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        word_a = st.selectbox("Positive word 1", WORDS, index=WORDS.index("king") if "king" in WORDS else 0)
    with c2:
        word_b = st.selectbox("Subtract word", WORDS, index=WORDS.index("man") if "man" in WORDS else 0)
    with c3:
        word_c = st.selectbox("Positive word 2", WORDS, index=WORDS.index("woman") if "woman" in WORDS else 0)

    analogy_vector = EMBEDDINGS[word_a] - EMBEDDINGS[word_b] + EMBEDDINGS[word_c]
    analogy_results = nearest_to_vector(
        analogy_vector,
        exclude_words={word_a, word_b, word_c},
        top_n=5
    )

    st.latex(rf"\text{{{word_a}}} - \text{{{word_b}}} + \text{{{word_c}}} \approx ?")
    st.subheader("Nearest words to the resulting vector")
    st.dataframe(analogy_results, use_container_width=True)

    st.write(
        "This works when semantic relationships correspond approximately to directions "
        "in the embedding space."
    )

# ------------------------------------------------------------
# Tab 3: Embedding Visualization
# ------------------------------------------------------------
with tab3:
    st.header("2D Embedding Visualization")
    st.write(
        "The original demo vectors live in 6 dimensions. PCA reduces them to 2 dimensions "
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

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(X_2d[:, 0], X_2d[:, 1])
        for i, word in enumerate(selected_words):
            ax.annotate(word, (X_2d[i, 0], X_2d[i, 1]), xytext=(5, 5), textcoords="offset points")
        ax.set_xlabel("First principal component")
        ax.set_ylabel("Second principal component")
        ax.set_title("PCA Projection of Word Embeddings")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.write("Explained variance ratio:")
        st.write(pd.DataFrame({
            "component": ["PC1", "PC2"],
            "explained_variance_ratio": pca.explained_variance_ratio_
        }))

# ------------------------------------------------------------
# Tab 4: Theory Notes
# ------------------------------------------------------------
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
        "In embedding spaces, similar directions often correspond to similar linguistic contexts."
    )

    st.subheader("Dimensionality Reduction")
    st.write(
        "High-dimensional embeddings are difficult to visualize directly. PCA and SVD "
        "can project vectors into lower-dimensional spaces while preserving important structure."
    )
    st.latex(r"A \approx U_k \Sigma_k V_k^T")

    st.subheader("Important Note")
    st.write(
        "This app uses intentionally simple demo vectors. In a full NLP system, embeddings "
        "would be learned from a large corpus using a method such as Word2Vec, GloVe, or a transformer model."
    )
