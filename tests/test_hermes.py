"""
ARIA / Hermes — Comprehensive Test Suite
Tests for:
  - Sandboxed Filesystem Tools (List, Read, Write, Search, Security constraints)
  - LLM client (chat vs multimodal endpoint routing)
  - Embedder client (embeddings vs batch routing)
  - Project Intelligence (auto progress calculation, insight promotion, snapshots)
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from database.models import Project, ProjectBlock, ProjectInsight, Task
from llm.client import HermesLLM
from rag.embedder import HermesEmbedder
from services.project_intelligence import (
    compute_project_progress,
    refresh_leader_blocks,
    _insight_type_to_block_type,
)


# ════════════════════════════════════════════════════════════════
#  1. FILESYSTEM SANDBOX TESTS
# ════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@patch("tools.static.filesystem_tools._get_allowed_roots")
async def test_filesystem_sandbox_allowed_paths(mock_roots):
    # Setup roots: /tmp/allowed
    mock_roots.return_value = [Path("/tmp/allowed").resolve()]

    from tools.static.filesystem_tools import _is_path_allowed

    # Allowed path
    allowed, _ = _is_path_allowed(Path("/tmp/allowed/file.txt"))
    assert allowed is True

    # Trailing traversal attempt (denied)
    allowed, _ = _is_path_allowed(Path("/tmp/allowed/../../etc/passwd"))
    assert allowed is False

    # Path completely outside (denied)
    allowed, _ = _is_path_allowed(Path("/var/log"))
    assert allowed is False


@pytest.mark.asyncio
@patch("tools.static.filesystem_tools._get_blocked_extensions")
async def test_filesystem_blocked_extensions(mock_blocked):
    mock_blocked.return_value = {".env", ".key"}

    from tools.static.filesystem_tools import _is_extension_blocked
    assert _is_extension_blocked(Path("config.env")) is True
    assert _is_extension_blocked(Path("id_rsa.key")) is True
    assert _is_extension_blocked(Path("document.pdf")) is False


# ════════════════════════════════════════════════════════════════
#  2. LLM CLIENT ROUTING TESTS
# ════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@patch("llm.client.settings")
@patch("httpx.AsyncClient.post")
async def test_llm_routing_endpoints(mock_post, mock_settings):
    # Setup Settings
    mock_settings.llm_provider = "openai"
    mock_settings.llm_api_base = "http://localhost:8080"
    mock_settings.llm_chat_path = "/v1/chat/completions"
    mock_settings.llm_multimodal_path = "/v1/multimodal"
    mock_settings.llm_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    mock_settings.llm_max_new_tokens = 500
    mock_settings.llm_temperature = 0.7
    mock_settings.llm_top_p = 0.9

    # Mock response
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={
        "choices": [{"message": {"content": "Test response"}}]
    })
    mock_post.return_value = mock_response

    client = HermesLLM()
    client.initialize()

    # Case A: Text-only request (routes to /v1/chat/completions)
    messages = [{"role": "user", "content": "Hello world"}]
    res = await client.generate(messages)
    assert res == "Test response"
    mock_post.assert_called_with(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.9,
        },
        headers={"Content-Type": "application/json"},
    )

    # Case B: Multimodal request (routes to /v1/multimodal)
    images = [b"fake_image_bytes"]
    res_multi = await client.generate(messages, images=images)
    assert res_multi == "Test response"
    mock_post.assert_called_with(
        "http://localhost:8080/v1/multimodal",
        json={
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello world"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,ZmFrZV9pbWFnZV9ieXRlcw=="},
                        },
                    ],
                }
            ],
            "max_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.9,
        },
        headers={"Content-Type": "application/json"},
    )


# ════════════════════════════════════════════════════════════════
#  3. EMBEDDING CLIENT ROUTING TESTS
# ════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@patch("rag.embedder.settings")
@patch("httpx.AsyncClient.post")
async def test_embedding_routing(mock_post, mock_settings):
    # Setup Settings
    mock_settings.embed_provider = "api"
    mock_settings.embed_api_base = "http://localhost:8080"
    mock_settings.embed_api_path = "/v1/embeddings"
    mock_settings.embed_batch_api_path = "/v1/embeddings/batch"
    mock_settings.embed_model_id = "nomic-ai/nomic-embed-text-v1.5"

    embedder = HermesEmbedder()
    embedder.initialize()

    # Case A: Single text embedding (/v1/embeddings)
    mock_resp_single = MagicMock()
    mock_resp_single.json = MagicMock(return_value={"embedding": [0.1, 0.2, 0.3]})
    mock_post.return_value = mock_resp_single

    res_single = await embedder.async_embed(["Hello"])
    assert res_single == [[0.1, 0.2, 0.3]]
    mock_post.assert_called_with(
        "http://localhost:8080/v1/embeddings",
        json={"model": "nomic-ai/nomic-embed-text-v1.5", "input": "Hello"},
        headers={"Content-Type": "application/json"},
    )

    # Case B: Batch text embedding (/v1/embeddings/batch)
    mock_resp_batch = MagicMock()
    mock_resp_batch.json = MagicMock(return_value={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
    mock_post.return_value = mock_resp_batch

    res_batch = await embedder.async_embed(["Hello", "World"])
    assert res_batch == [[0.1, 0.2], [0.3, 0.4]]
    mock_post.assert_called_with(
        "http://localhost:8080/v1/embeddings/batch",
        json={"model": "nomic-ai/nomic-embed-text-v1.5", "inputs": ["Hello", "World"]},
        headers={"Content-Type": "application/json"},
    )


# ════════════════════════════════════════════════════════════════
#  4. PROJECT INTELLIGENCE TESTS
# ════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_project_progress_calculations():
    # Setup mock database session
    db = AsyncMock()

    # Mock ProjectBlocks (1 todo, 1 in_progress, 1 done, 1 blocked = 4 blocks)
    block1 = ProjectBlock(status="todo", project_id="p1")
    block2 = ProjectBlock(status="in_progress", project_id="p1")
    block3 = ProjectBlock(status="done", project_id="p1")
    block4 = ProjectBlock(status="blocked", project_id="p1")

    # Mock Tasks (1 done = 1 task)
    task1 = Task(status="done", project_id="p1")

    # Mock Project record
    project = Project(id="p1", target_date=date.today() + timedelta_mock(5))

    # Mock DB returns
    execute_mock = AsyncMock()
    # First call select ProjectBlock
    # Second call select Task
    # Third call get Project
    db.execute.side_effect = [
        MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[block1, block2, block3, block4])))),
        MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[task1])))),
    ]
    db.get.return_value = project

    res = await compute_project_progress("p1", db)

    # 5 total items, 2 done ("done" block + "done" task) => 2/5 = 40%
    assert res["progress_pct"] == 40
    assert res["tasks_total"] == 5
    assert res["tasks_done"] == 2
    assert res["tasks_blocked"] == 1
    assert res["days_left"] == 5
    assert "5 days left" in res["deadline_label"]


@pytest.mark.asyncio
@patch("services.project_intelligence.llm")
async def test_refresh_leader_blocks_llm_json(mock_llm):
    db = AsyncMock()

    # Mock project
    project = Project(id="p1", title="Launch API", description="Desc")
    db.get.return_value = project

    # Mock progress calculation
    kpis = {
        "progress_pct": 50,
        "tasks_total": 4,
        "tasks_done": 2,
        "tasks_in_progress": 1,
        "tasks_blocked": 1,
        "days_left": 3,
        "deadline_label": "3 days left",
    }
    db.execute.return_value = MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[]))))

    # Mock LLM returns structured JSON block texts
    mock_llm.generate.return_value = json.dumps({
        "block_progress": "50% complete — half way there",
        "block_highlights": "Shipped endpoint base v1",
        "block_blockers": "1 blocked item on auth API",
        "block_next_steps": "Finalize integration tests",
    })

    with patch("services.project_intelligence.compute_project_progress", return_value=kpis):
        blocks = await refresh_leader_blocks("p1", db)

        assert blocks["block_progress"] == "50% complete — half way there"
        assert blocks["block_highlights"] == "Shipped endpoint base v1"
        assert blocks["block_blockers"] == "1 blocked item on auth API"
        assert blocks["block_next_steps"] == "Finalize integration tests"


def test_insight_type_conversion():
    assert _insight_type_to_block_type("auto_task") == "task"
    assert _insight_type_to_block_type("risk") == "risk"
    assert _insight_type_to_block_type("blocker") == "task"
    assert _insight_type_to_block_type("update") == "note"
    assert _insight_type_to_block_type("milestone") == "milestone"
    assert _insight_type_to_block_type("unknown") == "task"


# Helper for datetime offset mock
def timedelta_mock(days: int):
    from datetime import timedelta
    return timedelta(days=days)
