import numpy as np
from collections import defaultdict

# List of documents
documents = ["This is the first document.", 
             "This document is the second document.",
             "And this is the third one.", 
             "Is this the first document?"]

# Preprocessing: tokenize and build vocabulary
def preprocess_documents(docs):
    tokenized_docs = []
    vocab = set()
    for doc in docs:
        tokens = doc.lower().replace(".", "").replace("?", "").split()
        tokenized_docs.append(tokens)
        vocab.update(tokens)
    return tokenized_docs, sorted(vocab)

tokenized_docs, vocab = preprocess_documents(documents)
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

# Parameters
window_size = 2
embedding_dim = 10

# Build CBOW training data (context, target) pairs
def generate_cbow_data(tokenized_docs, window_size):
    data = []
    for tokens in tokenized_docs:
        for idx, target in enumerate(tokens):
            start = max(0, idx - window_size)
            end = min(len(tokens), idx + window_size + 1)
            context = [tokens[i] for i in range(start, end) if i != idx]
            if len(context) > 0:  # avoid empty context
                data.append((context, target))
    return data

cbow_data = generate_cbow_data(tokenized_docs, window_size)

# One-hot encoding function
def one_hot_vector(word, vocab):
    vec = np.zeros(len(vocab))
    vec[word2idx[word]] = 1
    return vec

# Initialize weights for input -> hidden and hidden -> output
W1 = np.random.randn(len(vocab), embedding_dim)
W2 = np.random.randn(embedding_dim, len(vocab))

# Simple CBOW forward pass
def cbow_forward(context_words):
    x = np.mean([one_hot_vector(w, vocab) for w in context_words], axis=0)  # average context vectors
    h = np.dot(x, W1)
    u = np.dot(h, W2)
    y_pred = softmax(u)
    return y_pred

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Example: print context-target pairs and forward pass output
for context, target in cbow_data[:5]:
    y_pred = cbow_forward(context)
    print(f"Context: {context} -> Target: {target} | Predicted distribution: {y_pred}")
