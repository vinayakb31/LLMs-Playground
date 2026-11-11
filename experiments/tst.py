from pathlib import Path

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "my_collection"


def print_chunks() -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    chunks = collection.get(include=["documents", "metadatas"])

    documents = chunks.get("documents") or []
    metadatas = chunks.get("metadatas") or []

    for index, document in enumerate(documents, start=1):
        source = metadatas[index - 1].get("source", "unknown")
        print(f"--- Chunk {index} | source: {source} ---")
        print(document)
        print()


if __name__ == "__main__":
    print_chunks()