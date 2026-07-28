"""
Long-term memory — stores past Q&A exchanges in a vector store (Chroma)
and retrieves semantically similar past interactions, ACROSS DIFFERENT
SESSIONS. This is distinct from short-term memory (Redis, see
redis_memory.py), which only recalls the current conversation's own
recent turns.

Example of the distinction:
  - Short-term: "What did I ask you two messages ago?" (same session)
  - Long-term: "Have I asked about nitrogen timing before?" (could be from
    a completely different session, days or weeks earlier)

Uses a general-purpose sentence embedding model (not Project 3's
biomedical-specific one — this stores conversational Q&A, not scientific
abstracts, so a general model is the right tool here, not a borrowed
domain-specific one).

Usage:
    from src.memory.long_term_memory import LongTermMemory
    memory = LongTermMemory()
    memory.add_interaction(
        question="What corn yield should I expect in Illinois?",
        answer="169.82 bu/acre...",
        session_id="session123",
    )
    relevant = memory.search_relevant_history("Tell me about Illinois yields", top_k=3)
"""

import logging
import time
import uuid

log = logging.getLogger(__name__)

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading general-purpose embedding model for long-term memory: "
                 "sentence-transformers/all-MiniLM-L6-v2 ...")
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder


class LongTermMemory:
    def __init__(self, persist_dir: str = "data/long_term_memory", collection_name: str = "conversations"):
        import chromadb

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_interaction(self, question: str, answer: str, session_id: str):
        """
        Embeds and stores a single Q&A exchange, tagged with the session
        it came from and a timestamp — searchable later regardless of
        which session is currently active.
        """
        embedder = _get_embedder()
        # Embed the question only (not the answer) — we're matching future
        # questions against past QUESTIONS primarily, since that's what a
        # user is likely to phrase similarly; the answer is stored as
        # metadata to return, not as part of the search key.
        embedding = embedder.encode([question])[0].tolist()

        entry_id = str(uuid.uuid4())
        self.collection.upsert(
            ids=[entry_id],
            embeddings=[embedding],
            documents=[question],
            metadatas=[{
                "answer": answer,
                "session_id": session_id,
                "timestamp": time.time(),
            }],
        )

    def search_relevant_history(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Returns up to top_k past Q&A exchanges (from ANY session) whose
        question is semantically similar to the current query. Returns an
        empty list if the memory store is empty (e.g. first-ever run) —
        this is expected, not an error.
        """
        if self.collection.count() == 0:
            return []

        embedder = _get_embedder()
        query_embedding = embedder.encode([query])[0].tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
        )

        history = []
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            history.append({
                "question": doc,
                "answer": metadata["answer"],
                "session_id": metadata["session_id"],
            })
        return history
