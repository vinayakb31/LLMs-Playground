import os
import instructor
import asyncio
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from groq import AsyncGroq
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from enum import Enum

load_dotenv()

client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

class user_request(BaseModel):
    input_str: str

SYSTEM_PROMPT = '''
You have an iq of 12 points.
Your responses are extremely dumb and often mostly unrelated to the question asked.
Give as useless and confusing responses as possible.
'''

async def generate_llm_output(input_str):
    stream = await client.chat.completions.create(
        model="openai/gpt-oss-20b",
        temperature=0.5,
        stream=True,
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":input_str},
        ]
    )
    
    async for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            payload = json.dumps({"token":token})
            yield f"data: {payload}\n\n"
    
    yield "data: [DONE]\n\n"

@app.post("/output/")
async def output(request: user_request):
    try:
        return StreamingResponse(
            generate_llm_output(request.input_str),
            media_type="text/plain"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))