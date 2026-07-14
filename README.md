# NLP-Word-Similarity-Explorer

# Linear Algebra in Modern NLP

This Streamlit application accompanies a Applied Linear Algebra final project, **Linear Algebra in Modern Natural Language Processing: Word Embeddings, Matrix Factorization, and Transformers**.

The goal of the app is to provide interactive demonstrations of several core linear algebra concepts used in natural language processing, including:

* Word embeddings as vectors
* Cosine similarity
* Vector analogies
* PCA-based embedding visualization
* Singular Value Decomposition (SVD)
* Low-rank matrix approximation

# Explore App

Below is the link to the interactive app.

[Streamlit App](https://nlp-word-similarity-explorer-3ib6astiybvefsgvecsdcy.streamlit.app/)

## Features

### Word Similarity Explorer

Select a word and view its nearest neighbors based on cosine similarity.

### Analogy Explorer

Experiment with vector arithmetic using examples such as:

**king − man + woman ≈ queen**

### Embedding Visualization

Project high-dimensional word vectors into two dimensions using PCA and visualize their geometric relationships.

### SVD Explorer

Explore low-rank approximations of a word co-occurrence matrix using truncated SVD. Adjust the rank parameter and observe:

* Matrix reconstruction
* Information retained
* Reconstruction error
* Singular values

## Technologies Used

* Python
* Streamlit
* NumPy
* Pandas
* Scikit-Learn
* Matplotlib
* Seaborn

## Running the App

Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

Then launch the Streamlit application:

```bash
streamlit run app.py
```

## Note

The embeddings used in this application are intentionally small demonstration vectors designed to illustrate linear algebra concepts. They are not pretrained production embeddings.

