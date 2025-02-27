import asyncio
import aiohttp
import time

import numpy as np

async def call_api(session, url, payload):
    async with session.post(url, json=payload) as response:
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
    ips = {
        'hal9004': '192.168.50.113',
        'hal9005': '192.168.50.153',
        'hal9006': '192.168.50.166'
    }

    urls = [
        f"http://{ip}:8000/nearest_neighbors" for ip in ips.values()
    ]
    
    # Common payload for all endpoints
    np.random.seed(0)
    embeddings = np.random.rand(3, 75264).tolist()
    payload = {
        "embeddings": embeddings,
        "n_neighbors": 10,
        "metric": "IP"
    }
    
    start_time = time.perf_counter()
    results = await call_all_apis(urls, payload)
    end_time = time.perf_counter()
    
    print(f"All requests completed in {end_time - start_time:.3f} seconds")
    
    dict_results = [result for result in results if isinstance(result, dict)]
    distances = np.concat([np.array(result['distances']) for result in dict_results], axis=1)
    embeddings = np.concat([np.array(result['nearest_embeddings']) for result in dict_results], axis=1)

    # I want to sort the embeddings by their distances
    sorted_indices = np.argsort(distances, axis=1)
    np.take_along_axis(embeddings, sorted_indices[..., None], axis=1)

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())

