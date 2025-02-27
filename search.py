import faiss
import numpy as np

# Load the embeddings and reshape them
embeddings_database = np.load('../comc_embeddings/swim_embeddings.npy')
dimensions = embeddings_database.shape[1]

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