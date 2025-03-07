import aiohttp
import asyncio
import time

import numpy as np

from pydantic import BaseModel
from fastapi import FastAPI

from image_utils import get_embeddings

async def post_url(session, url, payload):
    async with session.post(url, json=payload) as response:
        print(url, response.status)
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
    
async def main(embeddings: np.array, n_neighbors: int, n_cpus: int):
    t0 = time.perf_counter()
    # List of your API endpoints
    base_url = 'http://compute.hal9.com:{port}/nearest_neighbors'
    urls = [base_url.format(port=port) for port in ['8004', '8005', '8006']]

    # Common payload for all endpoints
    payload = {
        "embeddings": embeddings,
        "n_neighbors": n_neighbors,
        "metric": "IP",
        "n_cpus": n_cpus
    }
    
    start_time = time.perf_counter()
    results = await call_urls(*urls, payload=payload)
    end_time = time.perf_counter()
    
    print(f"All requests completed in {end_time - start_time:.3f} seconds")
    
    dict_results = [result for result in results if isinstance(result, dict)]
    distances = np.concatenate([result['distances'] for result in dict_results], axis=1)
    images = np.concatenate([result['images'] for result in dict_results], axis=1)

    # I want to sort the embeddings by their distances
    sorted_indices = np.argsort(distances, axis=1)
    sorted_images = np.take_along_axis(images, sorted_indices, axis=1)

    n_images = embeddings.shape[0]
    images_base_url = 'https://temphal9.s3.us-west-2.amazonaws.com/comc/data/0.0.1/extracted/'

    image_urls = np.char.add(images_base_url,  sorted_images[:,:n_neighbors].reshape(n_images*n_neighbors))    

    response = await call_urls(*image_urls)
    arr_response = np.array(response).reshape((n_images, n_neighbors))
    
    t1 = time.perf_counter()
    print(f'Time formatting and downloading images: {t1-end_time:.3f} seconds')
    print(f"Total time taken: {t1 - t0:.3f} seconds")

    return arr_response

class MainParams(BaseModel):
    encoded_images: list[str]
    n_neighbors: int
    n_cpus: int

app = FastAPI()

@app.get("/")
def root():
    return {"message": "COMC nearest neighbor search"}

@app.post("/nearest_neighbors")
def call_apis(params: MainParams):
    encoded_images = params.encoded_images
    n_neighbors = params.n_neighbors
    n_cpus = params.n_cpus
    
    embeddings = get_embeddings(encoded_images)

    return asyncio.run(main(embeddings, n_neighbors, n_cpus))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)