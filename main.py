import os
import instructor
from groq import AsyncGroq
from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

load_dotenv()

client = instructor.from_groq(
    AsyncGroq(api_key = os.getenv("GROQ_API_KEY")),
    mode=instructor.Mode.JSON
)

app = FastAPI()

class TicketPriority(str, Enum):
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'

class TicketCategory(str, Enum):
    BILLING = 'billing'
    BUG = 'bug'
    FEATURE_REQUEST = 'feature_request'
    OTHER = 'other'

class ExtractedTicket(BaseModel):
    summary: str = Field(description="1-sentence summary of the user issue.")
    category: TicketCategory = Field(description="Must match exactly: billing, bug, feature_request, or other.")
    priority: TicketPriority = Field(description="Must match exactly: low, medium, high, or critical.")
    action_required: bool = Field(description="True if an agent must act.")
    order_id: Optional[str] = Field(default=None, description="Order/invoice ID if mentioned.")
    sentiment_score: float = Field(ge=-1.0, le=1.0, description="-1.0 (angry) to 1.0 (happy).")
    
    @field_validator("category", mode="before")
    @classmethod
    
    def normalise_category(cls, value: str) -> str:
        if isinstance(value, str):
            value = value.strip().lower().replace(" ", "_")
        return value
    
    @field_validator("priority", mode="before")
    @classmethod

    def normalise_priority(cls, value: str) -> str:
        if isinstance(value, str):
            value = value.strip().lower()
        return value

SYSTEM_PROMPT = '''
You are an automated backend ticket parser.
You must NEVER respond conversationally.
You must ALWAYS return valid JSON matching the schema for any message.

Rules:
1. If the user message is just praise, greeting, or casual feedback, categorize it as 'other' with priority 'low' and action_required as false.
2. All enum values must be lowercase.

Examples:
Input: "Great App!"
Output: {
  "summary": "User expressed general appreciation for the app.",
  "category": "other",
  "priority": "low",
  "action_required": false,
  "order_id": null,
  "sentiment_score": 0.9
}
'''

class user_request(BaseModel):
    input_str: str

async def get_llm_output(input_str) -> ExtractedTicket:
    ticket: ExtractedTicket = await client.chat.completions.create(
        model = "openai/gpt-oss-20b",
        response_model = ExtractedTicket,
        max_retries=2,
        temperature=0.0,
        messages=[
            {"role":"system", "content":SYSTEM_PROMPT},
            {"role":"user", "content":input_str}
        ]
    )
    
    return ticket

@app.post("/output/", response_class=StreamingResponse)
async def output(request: user_request):
    try:
        llm_output = await get_llm_output(request.input_str)
        return llm_output
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))