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
        model = "openai/gpt-oss-120b",
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

queries = [
    # --- Self-Healing WAF: answerable ---
    "What is a web application firewall?",
    "What accuracy does the self-healing WAF achieve?",
    "Which datasets are used to train or evaluate the WAF?",
    "What role does OWASP Juice Shop play in the WAF project?",
    "How does the WAF defend against evasion attacks?",
    "How does the Hybrid Risk Engine make decisions?",
    "What is the inference latency of the WAF?",
    "Which Python file implements the hybrid risk engine?",
    "Which scripts are used in the WAF data augmentation pipeline?",
    "How does adversarial augmentation help the WAF?",

    # --- TurboQuant: answerable ---
    "What is the purpose of the TurboQuant project?",
    "How does TurboQuant support private, offline inference?",
    "What hardware does TurboQuant target?",
    "What software prerequisites are required to run TurboQuant?",
    "Does TurboQuant support persistent conversation history?",

    # --- Questions the corpus cannot answer ---
    "Who developed the TurboQuant algorithm?",
    "What GPT model does TurboQuant recommend?",
    "How much faster is TurboQuant than traditional quantisation?",
    "Can an RTX 3050 run TurboQuant reliably?",
    "What is the capital of France?",
]

generated_queries = {}

for query in queries:
    response = generate_multi_queries(query=query, client=groq_client)
    generated_queries[query] = response

print(generated_queries)