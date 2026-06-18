"""Tests for FastAPI endpoints."""

from fastapi.testclient import TestClient

from llmbrain.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "app" in data
    assert "version" in data

def test_observe_resource_endpoint():
    response = client.get("/observe/resource")
    assert response.status_code == 200
    data = response.json()
    assert "avg_cpu" in data
    assert "avg_mem" in data
    assert "snapshots" in data
    assert isinstance(data["snapshots"], list)

def test_observe_profiler_endpoint():
    response = client.get("/observe/profiler?top=5")
    assert response.status_code == 200
    data = response.json()
    assert "total_operations" in data
    assert "slowest" in data
    assert isinstance(data["slowest"], list)

def test_observe_queue_endpoint(tmp_path):
    response = client.get(f"/observe/queue?path={tmp_path}")
    assert response.status_code == 200
    data = response.json()
    assert "project_id" in data
    assert "queue_db" in data
    assert "stats" in data

def test_observe_semantic_search_endpoint(tmp_path):
    response = client.get(f"/observe/semantic-search?query=test&path={tmp_path}")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test"
    assert "results" in data
    assert isinstance(data["results"], list)
