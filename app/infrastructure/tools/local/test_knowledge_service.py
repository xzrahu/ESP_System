import asyncio
import httpx
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

async def test_knowledge_service():
    url = "http://127.0.0.1:8001/query"
    payload = {"question": "电脑不能开机怎么解决?"}
    
    print(f"Testing knowledge service at: {url}")
    print(f"Payload: {payload}")
    
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.post(
                url=url,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            print(f"Response status: {response.status_code}")
            print(f"Response content: {response.json()}")
            return response.json()
    except httpx.HTTPError as e:
        print(f"HTTP error: {str(e)}")
        return {"status": "error", "error_msg": f"HTTP error: {e}"}
    except Exception as e:
        print(f"General error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error_msg": f"General error: {e}"}

if __name__ == "__main__":
    result = asyncio.run(test_knowledge_service())
    print(f"Test result: {result}")