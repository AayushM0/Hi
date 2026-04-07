"""Tests for wikilink parser."""

import pytest
from lace.graph.parser import extract_wikilinks, extract_tags_as_links


def test_extract_single_wikilink():
    text = "Use [[asyncpg]] for database connections"
    links = extract_wikilinks(text)
    assert links == ["asyncpg"]


def test_extract_multiple_wikilinks():
    text = "See [[postgresql]] and [[alembic]] for migrations"
    links = extract_wikilinks(text)
    assert "postgresql" in links
    assert "alembic" in links


def test_extract_wikilink_normalizes_spaces():
    text = "See [[connection pooling]] for details"
    links = extract_wikilinks(text)
    assert "connection-pooling" in links


def test_extract_wikilink_normalizes_case():
    text = "Use [[AsyncPG]] driver"
    links = extract_wikilinks(text)
    assert "asyncpg" in links


def test_extract_no_wikilinks():
    text = "Plain text without any links"
    links = extract_wikilinks(text)
    assert links == []


def test_extract_tags_as_links():
    tags = ["asyncpg", "FastAPI", "db"]
    links = extract_tags_as_links(tags)
    assert "asyncpg" in links
    assert "fastapi" in links
    assert "db" in links


def test_extract_tags_empty():
    assert extract_tags_as_links([]) == []