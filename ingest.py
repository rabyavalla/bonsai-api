#!/usr/bin/env python3
"""
B.O.N.S.A.I. Pinecone Ingestion Script
========================================
Loads bonsai_agent_chunks_v3.json, generates embeddings via OpenAI,
and upserts into a Pinecone index.

Usage:
    python ingest.py --chunks ../bonsai_agent_chunks_v3.json

Requires:
    PINECONE_API_KEY and OPENAI_API_KEY in .env or environment
"""

import json
import argparse
import time
from pathlib import Path

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

from config import get_settings

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 96  # Pinecone upsert batch limit


def load_chunks(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    chunks = data["chunks"]
    print(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


def generate_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed texts in batches of 100."""
    all_embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch, dimensions=EMBEDDING_DIM)
        all_embeddings.extend([e.embedding for e in resp.data])
        print(f"  Embedded {min(i + 100, len(texts))}/{len(texts)}")
    return all_embeddings


def create_index_if_needed(pc: Pinecone, index_name: str):
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for ready
        while not pc.describe_index(index_name).status.get("ready"):
            time.sleep(1)
        print(f"Index '{index_name}' created and ready.")
    else:
        print(f"Index '{index_name}' already exists.")


def upsert_to_pinecone(pc: Pinecone, index_name: str, chunks: list[dict], embeddings: list[list[float]]):
    index = pc.Index(index_name)

    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        meta = chunk["metadata"]
        vectors.append(
            {
                "id": chunk["id"],
                "values": embedding,
                "metadata": {
                    "chunk_type": meta.get("chunk_type", ""),
                    "pillar": meta.get("pillar", ""),
                    "api_tier": meta.get("api_tier", ""),
                    "evidence_level": meta.get("evidence_level", ""),
                    "category": meta.get("category", ""),
                    "tags": meta.get("tags", []),
                    "clinical_references": meta.get("clinical_references", []),
                    "source_id": meta.get("source_id", ""),
                    "related_conditions": meta.get("related_conditions", []),
                    "content": chunk["content"][:8000],  # Pinecone metadata limit
                },
            }
        )

    # Upsert in batches
    total = 0
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i : i + BATCH_SIZE]
        index.upsert(vectors=batch)
        total += len(batch)
        print(f"  Upserted {total}/{len(vectors)}")

    print(f"\nDone. {total} vectors in '{index_name}'.")
    stats = index.describe_index_stats()
    print(f"Index stats: {stats.total_vector_count} vectors, dimension={stats.dimension}")


def main():
    parser = argparse.ArgumentParser(description="Ingest B.O.N.S.A.I. chunks into Pinecone")
    parser.add_argument("--chunks", default="data/bonsai_agent_chunks_v3.json", help="Path to chunks JSON")
    args = parser.parse_args()

    settings = get_settings()

    # Load chunks
    chunks = load_chunks(args.chunks)
    texts = [c["content"] for c in chunks]

    # Generate embeddings
    print("\nGenerating embeddings...")
    oai = OpenAI(api_key=settings.openai_api_key)
    embeddings = generate_embeddings(oai, texts)
    print(f"Generated {len(embeddings)} embeddings ({EMBEDDING_DIM}d)")

    # Create Pinecone index
    pc = Pinecone(api_key=settings.pinecone_api_key)
    create_index_if_needed(pc, settings.pinecone_index_name)

    # Upsert
    print("\nUpserting to Pinecone...")
    upsert_to_pinecone(pc, settings.pinecone_index_name, chunks, embeddings)


if __name__ == "__main__":
    main()
