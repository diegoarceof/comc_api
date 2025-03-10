import numpy as np

def save(embeddings: list[list[float]]):
    embeddings = np.array(embeddings)

    current = np.load("../../comc_embeddings", mmap_mode='r+')
    updated = np.concatenate((current, embeddings), axis = 0)

    np.save("../../comc_embeddings", axis = 0)