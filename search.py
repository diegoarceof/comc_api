import faiss
import numpy as np

# Load the embeddings and reshape them
embeddings_database = np.load('../comc_images/swim_embeddings.npy')
dimensions = embeddings_database.shape[1]

# Create the indexes
indexes = {
    'euclidean': faiss.IndexFlatL2(dimensions), 
    'IP': faiss.IndexFlatIP(dimensions)
}

for index in indexes.values():
    index.add(embeddings_database)

# Query the embeddings
def query(embeddings: np.array, n_neighbors: int, metric: str, n_cpus: int = 3):
    faiss.omp_set_num_threads(n_cpus)

    index = indexes[metric]
    distances, indices = index.search(embeddings, n_neighbors)
    
    return distances, embeddings_database[indices]