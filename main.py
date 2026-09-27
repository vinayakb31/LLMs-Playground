import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

stream = client.chat.completions.create(
    messages=[
        {
            "role":"system",
            "content":"you are a concise tech mentor. Give short and to the point answers only.",
            "role": "user",
            "content": "How temperature affects LLM outputs?",
        }
    ],

    temperature=0.5,
    stream=True,
    model="openai/gpt-oss-20b",
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end='', flush=True)