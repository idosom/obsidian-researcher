import json
import os
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


def read_markdown_files(vault_path: Path):
    md_files = []
    for root, dirs, files in os.walk(vault_path):
        dirs[:] = [d for d in dirs if d != ".obsidian" and not d.startswith(".")]
        for file in files:
            if file.lower().endswith(".md"):
                md_files.append(Path(root) / file)
    return md_files


def chunk_text(text: str, chunk_size: int, overlap: int):
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        if end < length:
            split_at = text.rfind("\n\n", start, end)
            if split_at == -1 or split_at <= start + chunk_size // 2:
                split_at = text.rfind("\n", start, end)
            if split_at != -1 and split_at > start:
                end = split_at

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= length:
            break

        start = max(end - overlap, start + 1)

    return chunks


def get_embedding(model: str, text: str):
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def main():
    config = load_config()

    vault_path = Path(config["vaultPath"])
    embedding_model = config["embeddingModel"]
    chunk_size = int(config["chunkSize"])
    chunk_overlap = int(config["chunkOverlap"])
    index_path = Path(config["indexPath"])
    chunks_path = Path(config["chunksPath"])

    if not vault_path.exists():
        raise FileNotFoundError(f"Vault path not found: {vault_path}")

    markdown_files = read_markdown_files(vault_path)
    if not markdown_files:
        raise RuntimeError("No markdown files found in the vault.")

    records = []
    embeddings = []

    for file_path in markdown_files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
            continue

        chunks = chunk_text(text, chunk_size, chunk_overlap)
        for i, chunk in enumerate(chunks):
            try:
                emb = get_embedding(embedding_model, chunk)
            except Exception as e:
                print(f"Embedding failed for {file_path} chunk {i}: {e}")
                continue

            records.append(
                {
                    "source": str(file_path),
                    "chunkIndex": i,
                    "text": chunk,
                }
            )
            embeddings.append(emb)

    if not embeddings:
        raise RuntimeError("No embeddings were created. Check your embedding model and vault content.")

    vectors = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    faiss.write_index(index, str(index_path))

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Indexed {len(records)} chunks from {len(markdown_files)} files.")
    print(f"Saved index to: {index_path}")
    print(f"Saved chunk metadata to: {chunks_path}")


if __name__ == "__main__":
    main()