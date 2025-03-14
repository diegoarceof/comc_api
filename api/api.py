import numpy as np
import uvicorn

from enum import Enum
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator
from typing import Optional

from search import query
from train import prepare, commit, rollback

app = FastAPI()

# Root endpoint to test connection
@app.get("/")
def root():
    return {"message": "COMC nearest neighbor search"}

# Define API parameters
class QueryParams(BaseModel):
    embeddings: list[list[float]]
    n_neighbors: int = 5
    n_cpus: int = 3
    database_name: str = 'COMC'

# Endpoint to query the database
@app.post("/nearest_neighbors")
async def nearest_neighbors(params: QueryParams):
    try:
        embeddings = np.array(params.embeddings)
        database_name = params.database_name

        n_neighbors = params.n_neighbors
        n_cpus = params.n_cpus

        distances, images = query(database_name, embeddings, n_neighbors, n_cpus)
        content = {'distances': distances.tolist(), 'images': images.tolist()}

        return JSONResponse(content=content)
    except Exception as e:
        return JSONResponse(content = {'error': str(e)}, status_code=400)

@app.get("/length")
async def get_length():
    return np.load('../../comc_images/image_names.npy', mmap_mode='r').shape[0]

# Define the saving parameters:
class PhaseEnum(str, Enum):
    prepare = "prepare"
    commit = "commit"
    rollback = "rollback"

class SaveParams(BaseModel):
    embeddings: Optional[list[list[float]]] = None
    names: Optional[list[str]] = None
    database_name: str = 'COMC'
    phase: PhaseEnum

    @model_validator(pre=True)
    def check_commit_params(cls, values):
        phase = values.get('phase')
        if phase == 'commit':
            if values.get('embeddings') is None:
                raise ValueError('embeddings must be provided for commit phase')
            if values.get('names') is None:
                raise ValueError('names must be provided for commit phase')
        return values

@app.post("/save_embeddings")
async def save_embeddings(params: SaveParams):
    # Lock the endpoint only for the database to be updated
    embeddings = params.embeddings
    names = params.names
    database_name = params.database_name
    phase = params.phase

    try:
        if phase == 'prepare':
            prepare(database_name)
        elif phase == 'commit':
            print(f'Saving {len(names)} new images')
            commit(database_name, embeddings, names)
        elif phase == "rollback":
            print('Rolling back previous changes')
            rollback(database_name)
        else:
            raise ValueError(f"phase {params.phase} is not supported.")
    except Exception as e:
        return JSONResponse(content = {'error': str(e)}, status_code=400)
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)