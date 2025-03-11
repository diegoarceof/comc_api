import numpy as np

def save(embeddings: list[list[float]], database_name: str):
    embeddings = np.array(embeddings)

    current = np.load("../../comc_embeddings", mmap_mode='r+')
    updated = np.concatenate((current, embeddings), axis = 0)

    print(f'Saving new embeddings on {database_name}. New total: {updated.shape[0] + current.shape[0]}')
    # np.save("../../comc_embeddings", axis = 0)