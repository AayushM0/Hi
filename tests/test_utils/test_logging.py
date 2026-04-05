"""Tests for retrieval logging."""

import json
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta

from lace.memory.models import make_memory, RetrievalResult
from lace.utils.logging import (
    RetrievalLogger,
    read_recent_logs,
    compute_retrieval_stats,
    compute_storage_stats,
    clean_old_logs,
)


@pytest.fixture
def lace_home(tmp_path):
    home = tmp_path / ".lace"
    (home / "logs" / "retrieval").mkdir(parents=True)
    (home / "logs" / "interactions").mkdir(parents=True)
    (home / "memory" / "vault").mkdir(parents=True)
    (home / "memory" / "vector_db").mkdir(parents=True)
    return home


def _make_result(content: str, score: float = 0.8) -> RetrievalResult:
    return RetrievalResult(
        memory=make_memory(content),
        relevance_score=score,
        match_type="vector",
        rank=1,
    )


def test_log_retrieval_creates_file(lace_home):
    logger  = RetrievalLogger(lace_home)
    results = [_make_result("Test memory")]
    logger.log_retrieval("test query", "global", results, 42.5)

    files = list((lace_home / "logs" / "retrieval").glob("*.jsonl"))
    assert len(files) == 1

    entry = json.loads(files[0].read_text().strip())
    assert entry["query"]         == "test query"
    assert entry["scope"]         == "global"
    assert entry["latency_ms"]    == 42.5
    assert entry["total_results"] == 1
    assert entry["type"]          == "retrieval"


def test_log_retrieval_zero_results(lace_home):
    logger = RetrievalLogger(lace_home)
    logger.log_retrieval("nothing found", "global", [], 10.0)

    entries = read_recent_logs(lace_home / "logs" / "retrieval", days=1)
    assert len(entries) == 1
    assert entries[0]["total_results"] == 0


def test_log_interaction(lace_home):
    logger = RetrievalLogger(lace_home)
    logger.log_interaction(
        query="how do I do X?",
        response_length=500,
        provider="ollama",
        model="llama3.2",
        memories_used=3,
        latency_ms=2500.0,
    )

    entries = read_recent_logs(
        lace_home / "logs" / "interactions",
        days=1,
        log_type="interaction",
    )
    assert len(entries) == 1
    assert entries[0]["provider"]      == "ollama"
    assert entries[0]["memories_used"] == 3


def test_read_recent_logs_empty(lace_home):
    entries = read_recent_logs(lace_home / "logs" / "retrieval", days=7)
    assert entries == []


def test_compute_retrieval_stats_empty(lace_home):
    stats = compute_retrieval_stats(lace_home / "logs" / "retrieval", days=7)
    assert stats["total_searches"] == 0
    assert stats["avg_latency_ms"] == 0.0


def test_compute_retrieval_stats_with_data(lace_home):
    logger = RetrievalLogger(lace_home)
    for i in range(5):
        results = [_make_result(f"Memory {i}", score=0.7 + i * 0.05)]
        logger.log_retrieval(
            f"query about database {i}", "global", results, 100.0 + i * 10
        )

    stats = compute_retrieval_stats(lace_home / "logs" / "retrieval", days=1)
    assert stats["total_searches"] == 5
    assert stats["avg_results"]    == 1.0
    assert stats["avg_latency_ms"] > 0
    assert stats["zero_result_rate"] == 0.0
    assert "database" in stats["top_queries"]


def test_zero_result_rate(lace_home):
    logger = RetrievalLogger(lace_home)
    logger.log_retrieval("found something", "global", [_make_result("x")], 50.0)
    logger.log_retrieval("found nothing",   "global", [],                  50.0)

    stats = compute_retrieval_stats(lace_home / "logs" / "retrieval", days=1)
    assert stats["total_searches"]   == 2
    assert stats["zero_result_rate"] == 50.0


def test_compute_storage_stats(lace_home):
    stats = compute_storage_stats(lace_home)
    assert "vault"     in stats
    assert "vector_db" in stats
    assert "logs"      in stats
    assert "total"     in stats


def test_clean_old_logs(lace_home):
    log_dir  = lace_home / "logs" / "retrieval"
    old_date = (datetime.now(timezone.utc) - timedelta(days=100)).strftime("%Y-%m-%d")
    today    = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    (log_dir / f"{old_date}.jsonl").write_text('{"type":"retrieval"}\n')
    (log_dir / f"{today}.jsonl").write_text('{"type":"retrieval"}\n')

    deleted = clean_old_logs(log_dir, retention_days=90)

    assert deleted == 1
    assert not (log_dir / f"{old_date}.jsonl").exists()
    assert (log_dir / f"{today}.jsonl").exists()


def test_multiple_log_entries(lace_home):
    logger = RetrievalLogger(lace_home)
    for i in range(10):
        logger.log_retrieval(f"query {i}", "global", [], 10.0 * i)

    entries = read_recent_logs(lace_home / "logs" / "retrieval", days=1)
    assert len(entries) == 10