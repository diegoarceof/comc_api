import asyncio
import aiohttp
import time

import numpy as np

async def call_api(session, url, payload):
    async with session.post(url, json=payload) as response:
        response_text = await response.text()
        print(url)
        print(f"Response status: {response.status}")
        print(f"Response content: {response_text}")
        return await response.json()

async def call_all_apis(urls, payload):
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(call_api(session, url, payload)) for url in urls
        ]
                
        # Wait for all tasks to complete and return results
        results = await asyncio.gather(*tasks)
        return results

async def main():
    # List of your API endpoints
    ports = ['8004', '8005', '8006']

    base_url = 'http://compute.hal9.com:{port}/nearest_neighbors'
    urls = [base_url.format(port=port) for port in ports]
    
    # Common payload for all endpoints
    np.random.seed(0)

    n_neighbors = 10
    n_images = 1

    embeddings = np.random.rand(n_images, 75264).tolist()
    payload = {
        "embeddings": embeddings,
        "n_neighbors": n_neighbors,
        "metric": "IP",
        "n_cpus": 4
    }
    
    start_time = time.perf_counter()
    results = await call_all_apis(urls, payload)
    end_time = time.perf_counter()
    
    print(f"All requests completed in {end_time - start_time:.3f} seconds")
    
    dict_results = [result for result in results if isinstance(result, dict)]
    distances = np.concatenate([np.array(result['distances']) for result in dict_results], axis=1)
    images = np.concatenate([np.array(result['images']) for result in dict_results], axis=1)

    # I want to sort the embeddings by their distances
    sorted_indices = np.argsort(distances, axis=1)
    sorted_images = np.take_along_axis(images, sorted_indices[..., None], axis=1)

    print(sorted_images[:,:n_neighbors,:].shape)

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
