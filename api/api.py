import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

from search import query
from search import save

# Define API parameters
class QueryParams(BaseModel):
    embeddings: list[list[float]]
    n_neighbors: int = 5
    n_cpus: int = 3
    database_name: str = 'COMC'

app = FastAPI()

# Root endpoint to test connection
@app.get("/")
def root():
    return {"message": "COMC nearest neighbor search"}

# Endpoint to query the database
@app.post("/nearest_neighbors")
async def nearest_neighbors(params: QueryParams):
    embeddings = np.array(params.embeddings)
    n_neighbors = params.n_neighbors
    
    n_cpus = params.n_cpus

    distances, images = query(embeddings, n_neighbors, n_cpus)
    return {
        'distances': distances.tolist(),
        'images': images.tolist()
        }

@app.get("/length")
async def get_length():
    return np.load('../comc_images/comc_embeddings.npy', mmap_mode='r').shape[0]

@app.post("/save_embeddings")
async def save_embeddings(embeddings: list[list[float]], database_name: str):
    try:
        save(embeddings)
        return {'content': 'Database updated succesfully'}, 200
    except Exception as e:
        return {'error': f'Error saving embeddings: {str(e)}'}, 400


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)