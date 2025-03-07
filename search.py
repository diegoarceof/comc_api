import faiss
import numpy as np

def create_index():
    embeddings_database = np.load('../comc_images/swim_embeddings.npy')
    embeddings_database /= np.linalg.norm(embeddings_database, keepdims=True)
    dimensions = embeddings_database.shape[1]

    # Create the index
    index = faiss.IndexFlatIP(dimensions)
    index.add(embeddings_database)

    return index

image_names = np.load('../comc_images/image_names.npy')
index = create_index()

# Query the embeddings
def query(embeddings: np.array, n_neighbors: int, n_cpus: int = 3):
    faiss.omp_set_num_threads(n_cpus)
    print(f'[Search] Initializing query search with {n_cpus} CPUs')

    distances, indices = index.search(embeddings, n_neighbors)
    names = image_names[indices]
    print(f'[Search] Found {n_neighbors} neighbors')

    print(f'[Search] First 3 indices {indices[:, :3]}')
    print(f'[Search] {distances = }')
    print(f'[Search] First three image names: {names[:,:3]}', end = '\n\n')

    return distances, names