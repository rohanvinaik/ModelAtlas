"""Shared test fixtures for ModelAtlas tests."""

from __future__ import annotations

import sqlite3

import pytest

from model_atlas import db


@pytest.fixture
def conn():
    """In-memory SQLite database with schema initialized."""
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    db.init_db(connection)
    yield connection
    connection.close()


@pytest.fixture
def populated_conn(conn):
    """Database pre-populated with sample models for query testing."""
    # Model 1: A standard 7B instruct model
    db.insert_model(conn, "meta-llama/Llama-3.1-8B-Instruct", author="meta-llama")
    db.set_position(
        conn, "meta-llama/Llama-3.1-8B-Instruct", "ARCHITECTURE", 0, 0, ["decoder-only"]
    )
    db.set_position(conn, "meta-llama/Llama-3.1-8B-Instruct", "EFFICIENCY", 0, 0)
    db.set_position(conn, "meta-llama/Llama-3.1-8B-Instruct", "CAPABILITY", 1, 2)
    db.set_position(conn, "meta-llama/Llama-3.1-8B-Instruct", "LINEAGE", 1, 1)
    db.set_position(conn, "meta-llama/Llama-3.1-8B-Instruct", "QUALITY", 1, 2)
    db.set_position(conn, "meta-llama/Llama-3.1-8B-Instruct", "DOMAIN", 0, 0)
    db.set_position(conn, "meta-llama/Llama-3.1-8B-Instruct", "COMPATIBILITY", 0, 0)
    for label in [
        "decoder-only",
        "instruction-following",
        "chat",
        "7B-class",
        "Llama-family",
        "community-favorite",
    ]:
        aid = db.get_or_create_anchor(conn, label, "CAPABILITY")
        db.link_anchor(conn, "meta-llama/Llama-3.1-8B-Instruct", aid)

    # Model 2: A small code model
    db.insert_model(conn, "Qwen/Qwen2.5-Coder-1.5B", author="Qwen")
    db.set_position(
        conn, "Qwen/Qwen2.5-Coder-1.5B", "ARCHITECTURE", 0, 0, ["decoder-only"]
    )
    db.set_position(conn, "Qwen/Qwen2.5-Coder-1.5B", "EFFICIENCY", -1, 2)
    db.set_position(conn, "Qwen/Qwen2.5-Coder-1.5B", "CAPABILITY", 1, 2)
    db.set_position(conn, "Qwen/Qwen2.5-Coder-1.5B", "LINEAGE", 0, 0)
    db.set_position(conn, "Qwen/Qwen2.5-Coder-1.5B", "QUALITY", 1, 1)
    db.set_position(conn, "Qwen/Qwen2.5-Coder-1.5B", "DOMAIN", 1, 1)
    db.set_position(conn, "Qwen/Qwen2.5-Coder-1.5B", "COMPATIBILITY", 0, 0)
    for label in [
        "decoder-only",
        "code-generation",
        "1B-class",
        "consumer-GPU-viable",
        "Qwen-family",
        "code-domain",
    ]:
        aid = db.get_or_create_anchor(conn, label, "CAPABILITY")
        db.link_anchor(conn, "Qwen/Qwen2.5-Coder-1.5B", aid)

    # Model 3: A GGUF quantized derivative
    db.insert_model(conn, "TheBloke/Llama-3.1-8B-Instruct-GGUF", author="TheBloke")
    db.set_position(conn, "TheBloke/Llama-3.1-8B-Instruct-GGUF", "ARCHITECTURE", 0, 0)
    db.set_position(conn, "TheBloke/Llama-3.1-8B-Instruct-GGUF", "EFFICIENCY", 0, 0)
    db.set_position(conn, "TheBloke/Llama-3.1-8B-Instruct-GGUF", "CAPABILITY", 1, 2)
    db.set_position(conn, "TheBloke/Llama-3.1-8B-Instruct-GGUF", "LINEAGE", 1, 3)
    db.set_position(conn, "TheBloke/Llama-3.1-8B-Instruct-GGUF", "QUALITY", 0, 0)
    db.set_position(conn, "TheBloke/Llama-3.1-8B-Instruct-GGUF", "DOMAIN", 0, 0)
    db.set_position(conn, "TheBloke/Llama-3.1-8B-Instruct-GGUF", "COMPATIBILITY", 1, 2)
    for label in [
        "decoder-only",
        "instruction-following",
        "GGUF-available",
        "Llama-family",
        "quantized",
    ]:
        aid = db.get_or_create_anchor(conn, label, "CAPABILITY")
        db.link_anchor(conn, "TheBloke/Llama-3.1-8B-Instruct-GGUF", aid)
    db.add_link(
        conn,
        "TheBloke/Llama-3.1-8B-Instruct-GGUF",
        "meta-llama/Llama-3.1-8B-Instruct",
        "quantized_from",
    )

    conn.commit()
    return conn
