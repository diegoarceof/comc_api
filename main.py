import asyncio
import aiohttp
import time

import numpy as np

async def post_url(session, url, payload):
    async with session.post(url, json=payload) as response:
        print(url, response.status)
        return await response.json()
    
async def get_url(session, url):
    async with session.get(url) as response:
        if response.headers.get('Content-Type', '').startswith('application/json'):
            return await response.json()
        else:
            # Handle other types of responses (e.g., image, text, etc.)
            return await response.read()  # Read the raw content

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
    
async def main(embeddings: list, n_neighbors: int):
    t0 = time.perf_counter()
    # List of your API endpoints
    base_url = 'http://compute.hal9.com:{port}/nearest_neighbors'
    urls = [base_url.format(port=port) for port in ['8004', '8005', '8006']]

    # Common payload for all endpoints
    payload = {
        "embeddings": embeddings,
        "n_neighbors": n_neighbors,
        "metric": "IP",
        "n_cpus": 4
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

    n_images = len(embeddings)
    images_base_url = 'https://temphal9.s3.us-west-2.amazonaws.com/comc/data/0.0.1/extracted/'

    image_urls = np.char.add(images_base_url,  sorted_images[:,:n_neighbors].reshape(n_images*n_neighbors))    

    response = await call_urls(*image_urls)
    # print(response)

    t1 = time.perf_counter()
    print(f"Total time taken: {t1 - t0:.3f} seconds")

# Run the async main function
if __name__ == "__main__":
    np.random.seed(0)

    n_images = 3
    n_neighbors = 5

    embeddings = np.random.rand(n_images, 75264).tolist()
    asyncio.run(main(embeddings, n_neighbors))
