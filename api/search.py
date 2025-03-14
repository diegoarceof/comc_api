import faiss
import numpy as np
from typing import Dict, Tuple

loaded_databases: Dict[str, Tuple] = {}

# Function to create index
def get_index(database_name: str) -> Tuple:
    global loaded_databases

    if database_name not in loaded_databases:
        # Load the database and normalize it. No need to reshape, it is already 2D
        embeddings_database = np.load(f'../../comc/{database_name}/swim_embeddings.npy')
        embeddings_database /= np.linalg.norm(embeddings_database, keepdims=True)
        dimensions = embeddings_database.shape[1]

        # Create the index
        index = faiss.IndexFlatIP(dimensions)
        index.add(embeddings_database)

        image_names = np.load('../../images/{database_name}/image_names.npy')
        loaded_databases[database_name] = (index, image_names)

    return loaded_databases[database_name]

# Function to query the embeddings
def query(database_name: str, embeddings: np.array, n_neighbors: int, n_cpus: int = 3):
    faiss.omp_set_num_threads(n_cpus)

    index, image_names = get_index(database_name)

    distances, indices = index.search(embeddings, n_neighbors)
    names = image_names[indices]

    return distances, names