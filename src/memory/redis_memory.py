"""
Redis-backed conversation memory — persists chat history per session so
the agent can maintain context across multiple turns, using Redis Cloud's
free tier (30MB, no credit card required).

Usage:
    from src.memory.redis_memory import SessionMemory
    memory = SessionMemory(session_id="user123")
    memory.add_message("user", "What's the yield forecast for Illinois?")
    memory.add_message("assistant", "...")
    history = memory.get_history()
"""

import json
import logging
import os

log = logging.getLogger(__name__)


class SessionMemory:
    """
    Stores conversation turns in Redis as a list under a session-specific
    key, with a TTL so idle sessions expire automatically (avoiding
    unbounded growth against the free tier's 30MB limit).
    """

    def __init__(self, session_id: str, ttl_seconds: int = 3600):
        import redis

        host = os.getenv("REDIS_HOST")
        port = int(os.getenv("REDIS_PORT", "6379"))
        password = os.getenv("REDIS_PASSWORD")

        if not host:
            raise RuntimeError(
                "REDIS_HOST environment variable not set. Set up a free Redis "
                "Cloud database at redis.io/cloud and export REDIS_HOST, "
                "REDIS_PORT, REDIS_PASSWORD — see README Setup section."
            )

        self.client = redis.Redis(
            host=host, port=port, password=password,
            decode_responses=True, ssl=True,
        )
        self.key = f"session:{session_id}:history"
        self.ttl_seconds = ttl_seconds

    def add_message(self, role: str, content: str):
        entry = json.dumps({"role": role, "content": content})
        self.client.rpush(self.key, entry)
        self.client.expire(self.key, self.ttl_seconds)  # refresh TTL on each turn

    def get_history(self) -> list[dict]:
        raw_entries = self.client.lrange(self.key, 0, -1)
        return [json.loads(e) for e in raw_entries]

    def clear(self):
        self.client.delete(self.key)
