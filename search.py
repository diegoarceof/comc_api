import faiss
import numpy as np

# Function to create index
def create_index():
    # Load the database and normalize it. No need to reshape, it is already 2D
    embeddings_database = np.load('../comc_images/swim_embeddings.npy')
    embeddings_database /= np.linalg.norm(embeddings_database, keepdims=True)
    dimensions = embeddings_database.shape[1]

    # Create the index
    index = faiss.IndexFlatIP(dimensions)
    index.add(embeddings_database)

    return index

# Load image names and index
image_names = np.load('../comc_images/image_names.npy')
index = create_index()

# Function to query the embeddings
def query(embeddings: np.array, n_neighbors: int, n_cpus: int = 3):
    faiss.omp_set_num_threads(n_cpus)

    distances, indices = index.search(embeddings, n_neighbors)
    names = image_names[indices]

    return distances, names