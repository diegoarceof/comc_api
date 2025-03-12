import numpy as np

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from search import query
from train import save

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
    embeddings = np.array(params.embeddings)
    n_neighbors = params.n_neighbors
    
    n_cpus = params.n_cpus

    distances, images = query(embeddings, n_neighbors, n_cpus)
    return JSONResponse(
        content={'distances': distances.tolist(), 'images': images.tolist()},
        status_code=200)

@app.get("/length")
async def get_length():
    return np.load('../../comc_images/image_names.npy', mmap_mode='r').shape[0]

# Define the saving parameters:
class SaveParams(BaseModel):
    embeddings: list[list[float]]
    database_name: str = 'COMC'

@app.post("/save_embeddings")
async def save_embeddings(params: SaveParams):
    print(f'Saving {len(params.embeddings)} on {params.database_name}')
    try:
        new_total = save(params.embeddings, params.database_name)
        print(f'{new_total} embeddings in the database')
        return JSONResponse(
            content=f"{params.database_name} database updated succesfully",
            status_code=200)
    except Exception as e:
        print(f'Error saving embeddings {str(e)}')
        return JSONResponse(
            content={"error": f"Error saving embeddings: {str(e)}"}, 
            status_code=400)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)