import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_multi_queries(query: str, client=groq_client):
    system_prompt = """
    You are a query generation assistant designed to improve
    document retrieval in a Retrieval-Augmented Generation (RAG) system.

    Generate exactly three alternative search queries based on
    the user's original question.

    Instructions:
    1. Preserve the original question's meaning and intent.
    2. Explore three different retrieval perspectives or aspects
    of the original question.
    3. Use relevant technical terminology, synonyms, and alternative
    expressions where appropriate.
    4. Each query should target a distinct perspective rather than
    simply rephrasing the same question.
    5. Do not introduce assumptions, unrelated concepts, or information
    that is not implied by the original question.
    6. Keep each query concise and suitable for semantic search.

    Return only the three queries, one per line.
    Do not include numbering, explanations, or additional text.
    """
    
    response = client.chat.completions.create(
        model = "openai/gpt-oss-20b",
        temperature = 0.2,
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": query
            }
        ]
    )
    
    response = response.choices[0].message.content
    response = response.split('\n')
    response = [i.strip() for i in response]
    
    return response

queries_to_test = [
    "What is the purpose of the TurboQuant project?",
    "Can I use TurboQuant without an internet connection?",
    "Does TurboQuant save previous conversations?"
]

generated_queries = {}

for query in queries_to_test:
    response = generate_multi_queries(query=query, client=groq_client)

    generated_queries[query] = response

print(generated_queries)