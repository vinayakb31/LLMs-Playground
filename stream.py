import os
import instructor
import asyncio
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

async def generate_llm_output(input_str):
    stream = await client.chat.completions.create(
        model="openai/gpt-oss-20b",
        temperature=0.5,
        stream=True,
        messages=[
            {"role":"system","content":"you are a modern philosopher"},
            {"role":"user","content":input_str},
        ]
    )
    
    async for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token

@app.post("/output/", response_class=StreamingResponse)
async def output(request: user_request):
    try:
        return StreamingResponse(
            generate_llm_output(request.input_str),
            media_type="text/plain"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))