import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

from search import query

# Define API parameters
class QueryParams(BaseModel):
    embeddings: list[list[float]]
    n_neighbors: int = 5
    n_cpus: int = 3

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)