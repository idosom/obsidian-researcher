import json
from pathlib import Path

import faiss
import numpy as np
import ollama

CONFIG_FILE = "config.json"


def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)

    required_keys = [
        "vaultPath",
        "llmModel",
        "embeddingModel",
        "topK",
        "chunkSize",
        "chunkOverlap",
        "indexPath",
        "chunksPath",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing config key: {key}")

    return config


def load_index_and_chunks(index_path: Path, chunks_path: Path):
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, chunks


def get_embedding(model: str, text: str):
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def search(index, chunks, embedding_model: str, query: str, top_k: int):
    query_vec = np.array([get_embedding(embedding_model, query)], dtype=np.float32)
    faiss.normalize_L2(query_vec)

    scores, ids = index.search(query_vec, top_k)

    results = []
    for idx in ids[0]:
        if idx == -1:
            continue
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])

    return results


def ask_question(config, index, chunks, question: str):
    matches = search(
        index=index,
        chunks=chunks,
        embedding_model=config["embeddingModel"],
        query=question,
        top_k=int(config["topK"]),
    )

    context_parts = []
    for i, item in enumerate(matches, start=1):
        context_parts.append(
            f"[Source {i}] {item['source']} (chunk {item['chunkIndex']}):\n{item['text']}"
        )

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found."

    response = ollama.chat(
        model=config["llmModel"],
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using only the provided notes when possible. "
                    "If the notes do not contain enough information, say that clearly. "
                    "Be concise and grounded in the retrieved context."
                ),
            },
            {
                "role": "user",
                "content": f"Notes:\n{context}\n\nQuestion: {question}",
            },
        ],
    )

    return response["message"]["content"], matches


def main():
    config = load_config()
    index_path = Path(config["indexPath"])
    chunks_path = Path(config["chunksPath"])

    index, chunks = load_index_and_chunks(index_path, chunks_path)

    print("Ready. Type a question, or 'exit' to quit.")

    while True:
        question = input("\nAsk: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        try:
            answer, matches = ask_question(config, index, chunks, question)
            print("\n" + answer)

            if matches:
                print("\nSources:")
                for item in matches:
                    print(f"- {item['source']} [chunk {item['chunkIndex']}]")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()