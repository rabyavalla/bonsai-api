"""
B.O.N.S.A.I. Retrieval Engine — Pinecone-backed
=================================================
Semantic search over 275 health intelligence chunks.
Used by Tier 1 (direct) and Tiers 2-4 (context retrieval).
"""

from typing import Optional

from openai import AsyncOpenAI
from pinecone import Pinecone

from config import get_settings

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


class BonsaiRetrieval:
    """Pinecone-backed retrieval for B.O.N.S.A.I. knowledge base."""

    def __init__(self):
        settings = get_settings()
        self._oai = AsyncOpenAI(api_key=settings.openai_api_key)
        pc = Pinecone(api_key=settings.pinecone_api_key)
        self._index = pc.Index(settings.pinecone_index_name)

    async def query(
        self,
        question: str,
        top_k: int = 3,
        pillar: Optional[str] = None,
        chunk_types: Optional[list[str]] = None,
        min_evidence: Optional[str] = None,
    ) -> list[dict]:
        """
        Semantic search over the B.O.N.S.A.I. knowledge base.

        Returns list of {content, metadata, score} dicts sorted by relevance.
        """
        # Embed the query
        resp = await self._oai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[question],
            dimensions=EMBEDDING_DIM,
        )
        query_vector = resp.data[0].embedding

        # Build Pinecone filter
        pc_filter = {}
        if pillar and pillar != "any":
            pc_filter["pillar"] = {"$eq": pillar}
        if chunk_types:
            pc_filter["chunk_type"] = {"$in": chunk_types}
        if min_evidence:
            evidence_order = {"strong": 3, "moderate": 2, "emerging": 1}
            min_level = evidence_order.get(min_evidence, 0)
            # Pinecone doesn't support custom ordinal filters natively,
            # so we filter the allowed levels
            allowed = [k for k, v in evidence_order.items() if v >= min_level]
            pc_filter["evidence_level"] = {"$in": allowed}

        # Query Pinecone
        results = self._index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=pc_filter if pc_filter else None,
        )

        # Format results
        formatted = []
        for match in results.matches:
            meta = match.metadata or {}
            formatted.append(
                {
                    "content": meta.get("content", ""),
                    "metadata": {
                        "chunk_type": meta.get("chunk_type"),
                        "pillar": meta.get("pillar"),
                        "evidence_level": meta.get("evidence_level"),
                        "category": meta.get("category"),
                        "tags": meta.get("tags", []),
                        "clinical_references": meta.get("clinical_references", []),
                        "source_id": meta.get("source_id"),
                        "related_conditions": meta.get("related_conditions", []),
                    },
                    "score": float(match.score),
                    "chunk_id": match.id,
                }
            )

        return formatted

    def format_tier1_response(self, results: list[dict], query: str, include_citations: bool = True) -> dict:
        """Format search results for Tier 1 API response."""
        formatted_results = []
        for r in results:
            entry = {
                "content": r["content"],
                "relevance_score": round(r["score"], 4),
                "pillar": r["metadata"]["pillar"],
                "evidence_level": r["metadata"].get("evidence_level", "not_rated"),
                "source_id": r["metadata"].get("source_id"),
            }
            if include_citations and r["metadata"].get("clinical_references"):
                entry["clinical_references"] = r["metadata"]["clinical_references"]
            if r["metadata"].get("related_conditions"):
                entry["related_conditions"] = r["metadata"]["related_conditions"]
            if r["metadata"].get("tags"):
                entry["tags"] = r["metadata"]["tags"]
            formatted_results.append(entry)

        return {
            "query": query,
            "results": formatted_results,
            "result_count": len(formatted_results),
            "dietary_philosophy": "WFPB",
            "framework": "B.O.N.S.A.I.",
            "disclaimer": (
                "Based on published clinical guidelines (ACLM, AHA, ADA, WHO) and "
                "peer-reviewed evidence. This information is educational and does not "
                "constitute medical advice. Not a substitute for individualized medical care."
            ),
        }
