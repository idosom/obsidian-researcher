# obsidian-Researcher

A local AI tool that lets you ask questions about your Obsidian vault and get answers based on your own notes.

It indexes your markdown files, stores them LOCALLY, retrieves the most relevant chunks, and uses a local Ollama model to answer. Nothing is sent to external APIs.

## Files

- `config.json` — vault path and model settings
- `index.py` — builds the local search index
- `main.py` — asks questions using the index
- `.gitignore` — keeps local files out of Git

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install ollama faiss-cpu numpy
```

Make sure Ollama is running locally, then pull the models you want to use.

## Configure

Edit `config.json`:

```json
{
  "vaultPath": "[FILEPATH]",
  "llmModel": "[MODEL]",
  "embeddingModel": "nomic-embed-text",
  "topK": 4,
  "chunkSize": 1800,
  "chunkOverlap": 250,
  "indexPath": "vault.index",
  "chunksPath": "chunks.json"
}
```

## Use

Build the index:

```bash
python index.py
```

Ask questions:

```bash
python main.py
```

```bash
Ask: What ideas have I written about startups?
```
