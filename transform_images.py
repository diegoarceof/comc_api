import os
import lmdb
import numpy as np
import pickle

# Save images to LMDB
def save_to_lmdb(images_array, lmdb_path):
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
def load_images_from_lmdb(lmdb_path, indices):
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

if __name__ == "__main__":
    # Load the images
    images = np.load('../comc_images/encoded_images.npy', allow_pickle=True)
    
    # Save the images to LMDB
    lmdb_path = '../comc_images/swim_images.lmdb'
    if not os.path.exists(lmdb_path):
        save_to_lmdb(images, lmdb_path) 
   
    # Load specific indices
    indices = np.random.randint(0, len(images), 10)
    loaded_images = load_images_from_lmdb(lmdb_path, indices)
    
    # Check if the loaded images are the same as the original images
    if not np.all(images[indices] == loaded_images):
        print(images[indices] == loaded_images)
    else:
        print("Images match")