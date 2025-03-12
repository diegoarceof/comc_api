import numpy as np

def save(embeddings: list[list[float]], database_name: str):
    embeddings = np.array(embeddings)

    current = np.load('../../comc_images/swim_embeddings.npy', mmap_mode='r+')
    updated = np.concatenate((current, embeddings), axis = 0)

    # np.save("../../comc_embeddings", axis = 0)
    return updated.shape[0] + current.shape[0]