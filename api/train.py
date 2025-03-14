import numpy as np

backup = {}
EMBEDDINGS_PATH = '../../comc_images/{database_name}/swim_embeddings.npy'
NAMES_PATH = '../../comc_images/{database_name}/image_names.npy'

def prepare(database_name: str):
    ## Functrion to load the embeddings and names. 
    backup[database_name] = (
        np.load(EMBEDDINGS_PATH.format(database_name), mmap_mode='r+'),
        np.load(NAMES_PATH.format(database_name), mmap_mode='r+')
    )

def commit(database_name: str, new_embeddings: list[list[float]], new_names: list[str]):    
    ## Function to save the previously loaded names and embeddings with new entries.
    embeddings, names = backup.get(database_name, (None, None))

    if (embeddings is None) or (names is None):
        raise ValueError(f'Database {database_name} is not prepared for updating.')

    embeddings_path = EMBEDDINGS_PATH.format(database_name)
    names_path = NAMES_PATH.format(database_name)

    np.save(embeddings_path, np.concatenate((embeddings, new_embeddings), axis=0))
    np.save(names_path, np.concatenate((names, new_names), axis = 0))

def rollback(database_name: str):
    ## Function to roll back any changes if the other APIs had an error.
    embeddings, names = backup.get(database_name, (None, None))

    if (embeddings is None) or (names is None):
        raise ValueError(f'Database {database_name} is not prepared for roll back.')

    embeddings_path = EMBEDDINGS_PATH.format(database_name)
    names_path = NAMES_PATH.format(database_name)

    np.save(embeddings_path, np.concatenate(embeddings, axis=0))
    np.save(names_path, np.concatenate(names, axis = 0))
