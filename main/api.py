import aiohttp
import asyncio
import time
import uvicorn

import numpy as np

from datetime import datetime, timezone
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Union, Dict, Optional

from model import get_embeddings

# Util functions for calling urls asynchronously
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

async def call_urls(urls, payload: Optional[Union[Dict, List[Dict]]] = None):    
    async with aiohttp.ClientSession() as session:
        if payload is None:
            tasks = [asyncio.create_task(get_url(session, url)) for url in urls]
        elif isinstance(payload, dict):
            tasks = [asyncio.create_task(post_url(session, url, payload)) for url in urls]
        elif isinstance(payload, list):
            tasks = [asyncio.create_task(post_url(session, url, p)) for url, p in zip(urls, payload)]
        else:
            raise ValueError('payload should be None, a dictionary or a list of dictionaries')
                
        # Wait for all tasks to complete and return results
        return await asyncio.gather(*tasks)

# IPs of the remote machines
IPS = [
        '192.168.50.113', # Hal9004 
        '192.168.50.153', # Hal9005
        '192.168.50.166'  # Hal9006
]        

# Initialize the app
app = FastAPI()
db_locks: Dict[str, asyncio.Lock] = {}

# Store the start time
start_time = datetime.now(timezone.utc)


########## ROOT ENDPOINT ##########
@app.get("/")
def root():
    content = {
        "message": "COMC nearest neighbor search",
        "up_since": start_time.isoformat()
    }
    return JSONResponse(content=content)


########## SEARCH ENDPOINT ##########
class SearchParams(BaseModel):
    database_name: str = 'COMC'
    n_neighbors: int = 10
    n_cpus: int = 7
    
    timestamp: float = datetime.now(timezone.utc).timestamp()

@app.post("/search")
async def search(files: List[UploadFile] = File(...), params: SearchParams = Form(...)):
    # Extract keys from the parameters
    database_name = params.database_name
    n_neighbors = params.n_neighbors
    n_cpus = params.n_cpus
    timestamp = params.timestamp

    transfer_time = datetime.now(timezone.utc).timestamp() - timestamp
    print(f'Finding {n_neighbors} neighbors for {len(files)} images')
    print(f'Transfer time: {transfer_time: .2f} seconds')

    t0 = time.perf_counter()
    embeddings = get_embeddings([await file.read() for file in files])
    
    t1 = time.perf_counter()
    urls = [f'http://{ip}:8000/nearest_neighbors' for ip in IPS]

    # Common payload for all endpoints
    payload = {
        "embeddings": embeddings.tolist(),
        "n_neighbors": n_neighbors,
        "n_cpus": n_cpus,
        "database_name": database_name
    }
   
    start_time = time.perf_counter()
    results = await call_urls(urls, payload)
    end_time = time.perf_counter()
    
    valid_results = [result for result in results if ('error' not in result)]

    distances = np.concatenate([result['distances'] for result in valid_results], axis=1)
    images = np.concatenate([result['images'] for result in valid_results], axis=1)
    
    # Sort distances in reverse 
    sorted_indices = np.argsort(-distances, axis=1)
    sorted_images = np.take_along_axis(images, sorted_indices, axis=1)

    t2 = time.perf_counter()
    print(f"All requests completed in {end_time - start_time:.3f} seconds")
    print(f'Formatting and sorting: {t2-end_time:.3f} seconds')
    print(f'Time calculating embeddings: {t1-t0:.3f}')
    print(f"Total query time taken: {t2 - t0:.3f} seconds", end = '\n\n')

    content = {
        'names': sorted_images.tolist(), 
        'transfer_time': transfer_time, 
        'timestamp': datetime.now(timezone.utc).timestamp()
    }

    return JSONResponse(content=content)


########## SAVE ENDPOINT ##########
def get_db_lock(database_name: str) -> asyncio.Lock:
    # Function to get the locks of each database to update
    if (database_name not in db_locks):
        db_locks[database_name] = asyncio.Lock()

    return db_locks[database_name]

@app.post("/save")
async def save(files: List[UploadFile] = File(...), database_name: str = Form('COMC')):
    save_urls = [f'http://{ip}:8000/save_embeddings' for ip in IPS]

    async with get_db_lock(database_name):
        try:
            ## Get the embeddings and names of the uploaded images
            embeddings = get_embeddings([await file.read() for file in files])
            embeddings = np.array_split(embeddings, len(IPS))

            names = [file.filename for file in files]
            names = np.array_split(names, len(IPS)) 

            ## Get the length of the database in each remote computer
            length_urls = [f'http://{ip}:8000/nearest_neighbors' for ip in IPS]
            lengths = await call_urls(length_urls) 
        except Exception as e:
            content = {
                'message': 'Unable to calculate to calculate embeddings or get database lengths.',
                'error': str(e)
            }
            return JSONResponse(content=content, status_code=400)

        try:    
            ## Call the first phase of the saving process: prepare
            responses = await call_urls(save_urls, payload = {'phase': 'prepare', 'database_name': database_name})
            if any(response.get('status_code') == 400 for response in responses):
                raise ValueError('Prepare phase not successful')
            
            ## Call the second phase of the saving process: commit
            commit_payloads = [{'embeddings': embeddings[ix].tolist(), 
                                'names': names[ix].tolist(), 
                                'database_name': database_name,
                                'phase': 'commit'} for ix in np.argsort(lengths)]
            
            responses = await call_urls(save_urls, commit_payloads)
            if any(response.get('status_code') == 400 for response in responses):
                raise ValueError('Commit phase not successful')
            
            return JSONResponse(content={'message': 'Embeddings saved successfully'})

        except Exception as e:
            # If an error arises on the first two steps, roll back any changes
            await call_urls(save_urls, {'database_name': database_name, 'phase': 'rollback'})

            return JSONResponse(content={'error': str(e)}, status_code=400)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)