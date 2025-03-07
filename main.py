import aiohttp
import asyncio
import time

import numpy as np

from datetime import datetime, timezone
from fastapi import FastAPI, File, UploadFile, Form
from typing import List

from image_utils import get_embeddings

async def post_url(session, url, payload):
    async with session.post(url, json=payload) as response:
        print(f'URL: {url}, {response.status}')
        return await response.json()
    
async def get_url(session, url):
    async with session.get(url) as response:
        if response.headers.get('Content-Type', '').startswith('application/json'):
            return await response.json()
        else:
            return await response.read() 

async def call_urls(*urls, payload = None):    
    async with aiohttp.ClientSession() as session:
        if payload is None:
            method = lambda url: get_url(session, url)
        else:
            method = lambda url: post_url(session, url, payload)

        tasks = [asyncio.create_task(method(url)) for url in urls]
                
        # Wait for all tasks to complete and return results
        results = await asyncio.gather(*tasks)
        return results

async def main(embeddings: np.array, n_neighbors: int, n_cpus: int, database_name: str):    
    t0 = time.perf_counter()

    ips = [
        '192.168.50.113', # Hal9004 
        '192.168.50.153', # Hal9005
        '192.168.50.166'  # Hal9006
    ]
    urls = [f'http://{ip}:8000/nearest_neighbors' for ip in ips]

    # Common payload for all endpoints
    payload = {
        "embeddings": embeddings.tolist(),
        "n_neighbors": n_neighbors,
        "n_cpus": n_cpus
    }
    
   
    start_time = time.perf_counter()
    results = await call_urls(*urls, payload=payload)
    end_time = time.perf_counter()
    
    print(f"All requests completed in {end_time - start_time:.3f} seconds")
    
    dict_results = [result for result in results if isinstance(result, dict)]
    distances = np.concatenate([result['distances'] for result in dict_results], axis=1)
    
    images = np.concatenate([result['images'] for result in dict_results], axis=1)
    
    # Sort distances in reverse 
    sorted_indices = np.argsort(-distances, axis=1, )

    sorted_images = np.take_along_axis(images, sorted_indices, axis=1)

    t1 = time.perf_counter()
    print(f'Formatting and sorting: {t1-end_time:.3f} seconds')
    print(f"Total query time taken: {t1 - t0:.3f} seconds", end = '\n\n')

    return sorted_images[:,:n_neighbors]

# Initialize the app
app = FastAPI()

# Root endpoint to test connection
@app.get("/")
def root():
    return {"message": "COMC nearest neighbor search"}

# Search endpoint to call each API
@app.post("/search")
async def upload_images(
        files: List[UploadFile] = File(...),
        database_name: str = Form('COMC'),
        n_neighbors: int = Form(10),
        n_cpus: int = Form(7),
        timestamp: float = Form(...)
    ):
    transfer_time = datetime.now(timezone.utc).timestamp() - timestamp
    print(f'Finding {n_neighbors} neighbors for {len(files)} images')
    print(f'Transfer time: {transfer_time: .2f} seconds')

    embeddings = get_embeddings([await file.read() for file in files])
    
    response = await main(embeddings, n_neighbors, n_cpus, database_name)
    return {'names': response.tolist(), 'transfer_time': transfer_time, 'timestamp': time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)