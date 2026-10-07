# LLMs Playground

A personal learning repository focused on exploring and building with modern AI tooling: RAG, vector databases, embeddings, LLM APIs, prompt engineering, structured outputs, and local experimentation with GenAI workflows.

This repo is not a polished production app—it's a practical playground for learning by building. Each script is a small experiment to understand how models, retrieval pipelines, and APIs fit together in real workflows.

## What this repo covers

- Retrieval-Augmented Generation (RAG)
- Vector databases and semantic search
- Chunking and embedding strategies
- LLM API integration
- Prompt design and structured outputs
- Streaming responses from models
- FastAPI + LLM experiments
- Working with local knowledge bases and sample documents

## Project structure

```text
.
├── main.py                  # Basic RAG pipeline using Chroma + sentence embeddings + Groq
├── ingest.py                # Ingests sample text files into a local vector store
├── search_retrieve.py       # Chunking + embedding + retrieval logic for sample documents
├── client.py                # FastAPI example for structured LLM output generation
├── stream.py                # Streaming response demo using server-sent events
├── Sample Data/             # Sample content used for retrieval experiments
├── chroma_db/              # Local persistent Chroma database
├── .env                    # Local environment variables (API keys)
├── debug_chunks.json       # Debug output for chunks
├── README.md               # Project documentation
└── .gitignore              # Repo ignore rules
```

## Core concepts explored

### 1. RAG pipeline
The repository includes a simple retrieval workflow where text is split into chunks, embedded, stored in Chroma, and then retrieved based on semantic similarity before generating a response using an LLM.

### 2. Vector DB and semantic search
Using Chroma as a local vector database, the project demonstrates how to:

- create a persistent collection
- generate embeddings for text chunks
- query by semantic similarity
- filter and rank results
- build a prompt from retrieved context

### 3. LLM API integration
Experiments connect to Groq-hosted models and use them for:

- question answering
- completion generation
- structure extraction
- streaming token responses

### 4. Structured outputs
The FastAPI examples show how to constrain model responses using Pydantic models and schema validation, which is useful for building agentic or backend-facing workflows.

## Example workflow

1. Read and preprocess documents
2. Split content into chunks
3. Embed chunks using a sentence transformer
4. Store them in Chroma
5. Search for semantically relevant chunks
6. Combine results into context
7. Ask an LLM to answer using the retrieved information

## Setup

This repo assumes Python 3.10+ and a local environment for experimentation.

### Install dependencies

```bash
pip install python-dotenv groq chromadb sentence-transformers fastapi uvicorn pydantic instructor httpx emoji
```

### Configure API key

Create a `.env` file in the repo root and add your API key:

```bash
GROQ_API_KEY=your_key_here
```

## Running the experiments

### Basic RAG example

```bash
python main.py
```

This runs a small RAG pipeline over sample queries.

### Ingest sample data

```bash
python ingest.py
```

This creates the local vector database entries from the documents in `Sample Data/`.

### Search and retrieve

```bash
python search_retrieve.py
```

This demonstrates chunk generation and retrieval logic.

### FastAPI structured output example

```bash
uvicorn ingest:app --reload
```

The file `ingest.py` defines a small FastAPI service that parses user requests into a structured ticket schema.

### Streaming LLM response example

```bash
uvicorn stream:app --reload
```

Then use the sample client:

```bash
python client.py
```

## Sample data

The repo includes example text documents for experimentation, including topics like:

- WAF / cybersecurity concepts
- hardware and AI compute notes
- knowledge-retrieval examples

These are meant to simulate a lightweight knowledge base for testing retrieval behavior.

## Learning goals

This project is meant to help learn and practice:

- how embeddings work in practice
- how vector search differs from keyword search
- how relevant context improves LLM responses
- how to build LLM-backed tools around structured schemas
- how to build local AI experiments without overengineering them

## Notes

- This is a learning playground, so code may be intentionally simple or exploratory.
- Local API keys and generated data are not meant for production use without review.
- Many scripts are meant to be run individually as experiments rather than as a single app.

## Future ideas

Some areas I'd like to explore next:

- adding a proper front-end demo for RAG
- comparing multiple embedding models
- trying more advanced chunking strategies
- adding hybrid search (keyword + vector)
- using LangChain / LlamaIndex or raw SDK patterns
- integrating Azure OpenAI or other model providers
- building a document Q&A app with user uploads

## License

This repository is for learning and experimentation. No formal license is currently specified.

## Acknowledgements

This project is inspired by hands-on exploration of:

- LLM application design
- retrieval augmentation
- vector databases
- semantic search
- AI product prototyping
