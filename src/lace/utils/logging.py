"""Retrieval and interaction logging for LACE.

Every search call is logged to JSONL files in ~/.lace/logs/.
This enables:
  - Retrieval quality measurement (precision, usage rate)
  - Latency tracking (p50, p95)
  - Usage patterns (what gets searched most)

Log format: one JSON object per line (JSONL).
Log rotation: one file per day.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lace.memory.models import RetrievalResult


# ── Log entry builders ────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_retrieval_log(
    query: str,
    scope: str,
    results: list[RetrievalResult],
    latency_ms: float,
    match_type: str = "vector",
) -> dict[str, Any]:
    """Build a retrieval log entry."""
    return {
        "type":          "retrieval",
        "timestamp":     _now_iso(),
        "query":         query,
        "scope":         scope,
        "match_type":    match_type,
        "latency_ms":    round(latency_ms, 2),
        "total_results": len(results),
        "results": [
            {
                "id":              r.memory.id,
                "summary":         r.memory.display_summary(),
                "category":        r.memory.category.value,
                "relevance_score": r.relevance_score,
                "confidence":      r.memory.confidence,
                "rank":            r.rank,
            }
            for r in results
        ],
    }


def _make_interaction_log(
    query: str,
    response_length: int,
    provider: str,
    model: str,
    memories_used: int,
    latency_ms: float,
) -> dict[str, Any]:
    """Build an interaction log entry (for lace ask)."""
    return {
        "type":            "interaction",
        "timestamp":       _now_iso(),
        "query":           query,
        "provider":        provider,
        "model":           model,
        "memories_used":   memories_used,
        "response_length": response_length,
        "latency_ms":      round(latency_ms, 2),
    }


# ── Log writer ────────────────────────────────────────────────────────────────

class RetrievalLogger:
    """Writes retrieval and interaction logs to JSONL files.

    One file per day:
      ~/.lace/logs/retrieval/2025-01-15.jsonl
      ~/.lace/logs/interactions/2025-01-15.jsonl
    """

    def __init__(self, lace_home: Path) -> None:
        self.lace_home = lace_home
        self.retrieval_dir = lace_home / "logs" / "retrieval"
        self.interaction_dir = lace_home / "logs" / "interactions"

        self.retrieval_dir.mkdir(parents=True, exist_ok=True)
        self.interaction_dir.mkdir(parents=True, exist_ok=True)

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _retrieval_log_path(self) -> Path:
        return self.retrieval_dir / f"{self._today()}.jsonl"

    def _interaction_log_path(self) -> Path:
        return self.interaction_dir / f"{self._today()}.jsonl"

    def _append(self, path: Path, entry: dict[str, Any]) -> None:
        """Append a JSON entry to a JSONL file. Silent on failure."""
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass  # Logging must never break the main flow

    def log_retrieval(
        self,
        query: str,
        scope: str,
        results: list[RetrievalResult],
        latency_ms: float,
        match_type: str = "vector",
    ) -> None:
        """Log a retrieval operation."""
        entry = _make_retrieval_log(
            query=query,
            scope=scope,
            results=results,
            latency_ms=latency_ms,
            match_type=match_type,
        )
        self._append(self._retrieval_log_path(), entry)

    def log_interaction(
        self,
        query: str,
        response_length: int,
        provider: str,
        model: str,
        memories_used: int,
        latency_ms: float,
    ) -> None:
        """Log a lace ask interaction."""
        entry = _make_interaction_log(
            query=query,
            response_length=response_length,
            provider=provider,
            model=model,
            memories_used=memories_used,
            latency_ms=latency_ms,
        )
        self._append(self._interaction_log_path(), entry)


# ── Log reader ────────────────────────────────────────────────────────────────

def read_recent_logs(
    log_dir: Path,
    days: int = 7,
    log_type: str = "retrieval",
) -> list[dict[str, Any]]:
    """Read log entries from the last N days.

    Returns entries newest first.
    """
    from datetime import timedelta

    entries: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)

    for i in range(days):
        date = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        log_file = log_dir / f"{date}.jsonl"

        if not log_file.exists():
            continue

        try:
            with open(log_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            if entry.get("type") == log_type:
                                entries.append(entry)
                        except json.JSONDecodeError:
                            pass
        except Exception:
            pass

    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return entries


# ── Analytics ─────────────────────────────────────────────────────────────────

def compute_retrieval_stats(
    log_dir: Path,
    days: int = 7,
) -> dict[str, Any]:
    """Compute retrieval quality metrics from logs."""
    entries = read_recent_logs(log_dir, days=days, log_type="retrieval")

    if not entries:
        return {
            "total_searches":      0,
            "avg_results":         0.0,
            "avg_latency_ms":      0.0,
            "p95_latency_ms":      0.0,
            "avg_relevance_score": 0.0,
            "zero_result_rate":    0.0,
            "top_queries":         [],
            "days_analyzed":       days,
        }

    latencies     = [e["latency_ms"] for e in entries if "latency_ms" in e]
    result_counts = [e["total_results"] for e in entries if "total_results" in e]
    zero_results  = sum(1 for c in result_counts if c == 0)

    # Top relevance scores (first result per search)
    top_scores: list[float] = []
    for e in entries:
        results = e.get("results", [])
        if results:
            top_scores.append(results[0].get("relevance_score", 0))

    # Query word frequency
    from collections import Counter
    query_words: list[str] = []
    for e in entries:
        words = [w.lower() for w in e.get("query", "").split() if len(w) > 3]
        query_words.extend(words)

    top_queries = [word for word, _ in Counter(query_words).most_common(5)]

    # P95 latency
    sorted_latencies = sorted(latencies)
    p95_idx = min(int(len(sorted_latencies) * 0.95), len(sorted_latencies) - 1)
    p95_latency = sorted_latencies[p95_idx] if sorted_latencies else 0.0

    return {
        "total_searches":      len(entries),
        "avg_results":         round(sum(result_counts) / len(result_counts), 1) if result_counts else 0,
        "avg_latency_ms":      round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "p95_latency_ms":      round(p95_latency, 1),
        "avg_relevance_score": round(sum(top_scores) / len(top_scores), 3) if top_scores else 0,
        "zero_result_rate":    round(zero_results / len(entries) * 100, 1),
        "top_queries":         top_queries,
        "days_analyzed":       days,
    }


def compute_storage_stats(lace_home: Path) -> dict[str, str]:
    """Compute storage usage for vault, vector DB, and logs."""

    def dir_size_mb(path: Path) -> float:
        if not path.exists():
            return 0.0
        total = sum(
            f.stat().st_size
            for f in path.rglob("*")
            if f.is_file()
        )
        return round(total / (1024 * 1024), 2)

    vault_mb  = dir_size_mb(lace_home / "memory" / "vault")
    vector_mb = dir_size_mb(lace_home / "memory" / "vector_db")
    logs_mb   = dir_size_mb(lace_home / "logs")

    def fmt(mb: float) -> str:
        if mb < 1:
            return f"{int(mb * 1024)}KB"
        return f"{mb}MB"

    return {
        "vault":     fmt(vault_mb),
        "vector_db": fmt(vector_mb),
        "logs":      fmt(logs_mb),
        "total":     fmt(vault_mb + vector_mb + logs_mb),
    }


def clean_old_logs(log_dir: Path, retention_days: int = 90) -> int:
    """Delete log files older than retention_days.

    Returns number of files deleted.
    """
    from datetime import timedelta

    now     = datetime.now(timezone.utc)
    cutoff  = now - timedelta(days=retention_days)
    deleted = 0

    for log_file in log_dir.glob("*.jsonl"):
        try:
            file_date = datetime.strptime(
                log_file.stem, "%Y-%m-%d"
            ).replace(tzinfo=timezone.utc)
            if file_date < cutoff:
                log_file.unlink()
                deleted += 1
        except (ValueError, OSError):
            pass

    return deleted