import httpx
import asyncio
import json

API_URL = "http://127.0.0.1:8000/output/"

async def stream_client(prompt: str):
    payload = {"input_str": prompt}
    
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", API_URL, json=payload) as response:
            if response.status_code != 200:
                print(f"Server error: {response.status_code}")
                return
            
            async for line in response.aiter_lines():
                line = line.strip()
                
                if not line or not line.startswith("data: "):
                    continue
                
                payload_str = line[6:]
                
                if payload_str == "[DONE]":
                    print("\n\n-----Stream Completed-----")
                    break

                try:
                    data = json.loads(payload_str)
                    
                    if "error" in data:
                        print("\n\nError: {data['error']}")
                        break

                    if "token" in data:
                        print(data['token'], end='', flush=True)
                
                except json.JSONDecodeError:
                    print(payload_str, end='', flush=True)
                    
if __name__ == "__main__":
    prompt_text = str(input("Query: "))
    print("\n\nResponse: ", end="")
    asyncio.run(stream_client(prompt_text))