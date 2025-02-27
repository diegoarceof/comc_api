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
    embeddings = np.random.rand(1, 75264).tolist()
    payload = {
        "embeddings": embeddings,
        "n_neighbors": 1,
        "metric": "IP"
    }
    
    start_time = time.time()
    results = await call_all_apis(urls, payload)
    end_time = time.time()
    
    print(f"All requests completed in {end_time - start_time:.2f} seconds")
    
    dict_results = [result for result in results if isinstance(result, dict)]
    for result in dict_results:
        print(np.array(result['distances']).shape, np.array(result['nearest_embeddings']).shape)

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())

