import httpx
import asyncio

async def test():
    # Correct: json={"input_str": "..."}
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST", 
            "http://127.0.0.1:8000/output/", 
            json={"input_str": "Why do humans seek meaning?"}
        ) as res:
            async for line in res.aiter_lines():
                print(line)

asyncio.run(test())