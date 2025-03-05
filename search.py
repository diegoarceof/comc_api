import faiss
import lmdb
import numpy as np
import pickle

lmdb_path = '../comc_images/swim_images.lmdb'

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

# Save images to LMDB
def save_to_lmdb(images_array):
    # Create LMDB environment
    env = lmdb.open(lmdb_path, map_size=int(1e10))  # Adjust map_size based on your data size
    
    # Write data
    with env.begin(write=True) as txn:
        for i, img in enumerate(images_array):
            # Convert index to byte key
            key = f"{i:08d}".encode()
            # Save the image
            txn.put(key, pickle.dumps(img))
    
    env.close()

# Load specific indices
def load_images_from_lmdb(indices):
    # Open LMDB environment
    env = lmdb.open(lmdb_path, readonly=True)
    
    results = {}
    # Read data
    with env.begin() as txn:
        for idx in indices:
            # Convert index to byte key
            key = f"{idx:08d}".encode()
            # Get and deserialize the image
            img_bytes = txn.get(key)
            if img_bytes is not None:
                results[idx] = pickle.loads(img_bytes)
    
    env.close()
    
    # Return in the order of indices
    return [results[idx] for idx in indices if idx in results]

# Query the embeddings
def query(embeddings: np.array, n_neighbors: int, metric: str, n_cpus: int = 3):
    faiss.omp_set_num_threads(n_cpus)

    index = indexes[metric]
    distances, indices = index.search(embeddings, n_neighbors)
    images_data = [load_images_from_lmdb(idxs) for idxs in indices]
    
    print('Search successful')
    
    return distances, images_data
