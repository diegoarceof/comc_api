import faiss
import numpy as np

# Load the embeddings and reshape them
dimensions = 75264
embeddings_database = np.load('embeddings/swim_embeddings.npy').reshape(-1, dimensions)

# Create the indexes
indexes = {
    'euclidean': faiss.IndexFlatL2(dimensions), 
    'IP': faiss.IndexFlatIP(dimensions)
}

for index in indexes:
    indexes[index].add(embeddings_database)

# Query the embeddings
def query(embeddings, n_neighbors, metric):
    index = indexes[metric]
    distances, indices = index.search(embeddings, n_neighbors)
    
    return distances, embeddings_database[indices]