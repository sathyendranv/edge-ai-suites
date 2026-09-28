# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""UI service tests — uses HTTPX respx to mock backend services."""

import os
import pytest

os.environ["MQTT_DISABLED"] = "true"
os.environ["AGENT_SERVICE_URL"]     = "http://mock-agent"
os.environ["DETECTION_SERVICE_URL"] = "http://mock-detection"
os.environ["STORAGE_SERVICE_URL"]   = "http://mock-storage"
os.environ["USE_CASE_ID"]           = "test-case"

import respx
import httpx
from fastapi.testclient import TestClient
from src.app import app
from src import workbench


def assert_condition(condition, message=""):
    """Fail a test when condition is false."""
    assert condition, message  # nosec B101


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@respx.mock
def test_index_no_data(client):
    respx.get("http://mock-storage/detections/summary").mock(return_value=httpx.Response(200, json={}))
    respx.get("http://mock-detection/detection/runs").mock(return_value=httpx.Response(200, json=[]))
    respx.get("http://mock-agent/agents/runs").mock(return_value=httpx.Response(200, json=[]))
    respx.get("http://mock-detection/detection/videos").mock(return_value=httpx.Response(200, json={"videos": []}))
    r = client.get("/")
    assert_condition(r.status_code == 200)
    assert_condition("Agentic Weld Quality Analysis" in r.text)


@respx.mock
def test_index_with_summary(client):
    summary = {
        "by_class": [
            {"label": "Rupture", "count": 5, "avg_confidence": 0.88, "max_confidence": 0.95}
        ]
    }
    respx.get("http://mock-storage/detections/summary").mock(return_value=httpx.Response(200, json=summary))
    respx.get("http://mock-detection/detection/runs").mock(return_value=httpx.Response(200, json=[]))
    respx.get("http://mock-agent/agents/runs").mock(return_value=httpx.Response(200, json=[]))
    respx.get("http://mock-detection/detection/videos").mock(return_value=httpx.Response(200, json={"videos": []}))
    r = client.get("/")
    assert_condition(r.status_code == 200)
    assert_condition("Rupture" in r.text)


@respx.mock
def test_index_merges_detection_and_agent_runs(client):
    respx.get("http://mock-storage/detections/summary").mock(return_value=httpx.Response(200, json={}))
    respx.get("http://mock-detection/detection/videos").mock(return_value=httpx.Response(200, json={"videos": []}))
    respx.get("http://mock-detection/detection/runs").mock(return_value=httpx.Response(200, json=[
        {"run_id": "r1", "status": "completed", "phase": "completed", "result": {}},
        {"run_id": "r2", "status": "running", "phase": "detecting", "result": None},
    ]))
    respx.get("http://mock-agent/agents/runs").mock(return_value=httpx.Response(200, json=[
        {"run_id": "r1", "status": "completed", "phase": "completed"},
    ]))
    r = client.get("/")
    assert_condition(r.status_code == 200)
    assert_condition("r1"[:8] in r.text or "r1" in r.text)


@respx.mock
def test_detections_page(client):
    detections = [
        {"frame_id": 1, "label": "Rupture", "confidence": 0.9, "x": 10, "y": 10, "width": 50, "height": 40, "timestamp": "2026-01-01T00:00:00"}
    ]
    respx.get("http://mock-storage/detections").mock(return_value=httpx.Response(200, json=detections))
    r = client.get("/detections")
    assert_condition(r.status_code == 200)
    assert_condition("Rupture" in r.text)


def test_health(client):
    r = client.get("/health")
    assert_condition(r.status_code == 200)
    assert_condition(r.json()["service"] == "ui-service")
    assert_condition(r.json()["use_case_id"] == "test-case")


def test_insights_page(client):
    r = client.get("/insights")
    assert_condition(r.status_code == 200)
    assert_condition("Insights Workbench" in r.text)
    assert_condition("const APP_BASE_PATH = \"/insights\"" in r.text)


def test_insights_measurements_api(client):
    r = client.get("/insights/api/measurements")
    assert_condition(r.status_code == 200)
    assert_condition(r.json()["measurements"] == ["fusion_result"])


def test_dashboard_nav_includes_insights(client):
    with respx.mock:
        respx.get("http://mock-storage/detections/summary").mock(return_value=httpx.Response(200, json={}))
        respx.get("http://mock-agent/agents/runs").mock(return_value=httpx.Response(200, json=[]))
        respx.get("http://mock-detection/detection/videos").mock(return_value=httpx.Response(200, json={"videos": []}))
        r = client.get("/")
    assert_condition(r.status_code == 200)
    assert_condition("Insights Workbench" in r.text)


def test_insights_data_api(client, monkeypatch):
    class FakeQueryResult:
        def __init__(self, points):
            self._points = points

        def get_points(self):
            return iter(self._points)

    class FakeInfluxClient:
        def query(self, query):
            assert "fusion_result" in query
            return FakeQueryResult(
                [
                    {
                        "time": "2026-01-01T00:00:00Z",
                        "timeseries_classification": "good",
                        "vision_classification": "good",
                        "fused_decision": "good",
                    }
                ]
            )

    monkeypatch.setattr(workbench, "_get_influx_client", lambda: FakeInfluxClient())

    r = client.get("/insights/api/data?page=1&page_size=10")
    assert_condition(r.status_code == 200)
    payload = r.json()
    assert_condition(payload["has_more"] is False)
    assert_condition(payload["rows"][0]["fused_decision"] == "good")


def test_insights_data_api_normalizes_pagination(client, monkeypatch):
    class FakeQueryResult:
        def get_points(self):
            return iter([])

    class FakeInfluxClient:
        def query(self, query):
            assert "LIMIT 201 OFFSET 0" in query
            return FakeQueryResult()

    monkeypatch.setattr(workbench, "_get_influx_client", lambda: FakeInfluxClient())

    r = client.get("/insights/api/data?page=0&page_size=999")
    assert_condition(r.status_code == 200)
    payload = r.json()
    assert_condition(payload["page"] == 1)
    assert_condition(payload["page_size"] == 200)


def test_insights_data_api_handles_influx_error(client, monkeypatch):
    def raise_influx_error():
        raise RuntimeError("boom")

    monkeypatch.setattr(workbench, "_get_influx_client", raise_influx_error)

    r = client.get("/insights/api/data")
    assert_condition(r.status_code == 500)
    assert_condition(r.json()["error"] == "Unable to load data")


def test_insights_explain_rejects_invalid_time(client, monkeypatch):
    monkeypatch.setattr(workbench, "_get_influx_client", lambda: object())

    r = client.post("/insights/api/explain", json={"selected_times": ["not-a-time"]})
    assert_condition(r.status_code == 400)
    assert_condition("Invalid time format" in r.json()["error"])


def test_insights_explain_handles_vllm_error(client, monkeypatch):
    class FakeQueryResult:
        def __init__(self, points):
            self._points = points

        def get_points(self):
            return iter(self._points)

    class FakeInfluxClient:
        def query(self, query):
            if "vision-weld-classification-results" in query:
                return FakeQueryResult([])
            if "weld-sensor-anomaly-data" in query:
                return FakeQueryResult(
                    [
                        {
                            "Primary Weld Current": 10,
                            "Secondary Weld Voltage": 20,
                            "Pressure": 30,
                            "CO2 Weld Flow": 40,
                            "Feed": 50,
                            "Wire Consumed": 60,
                        }
                    ]
                )
            return FakeQueryResult(
                [
                    {
                        "vision_timestamp": "2026-01-01T00:00:00Z",
                        "timeseries_timestamp": "1704067200000000000",
                    }
                ]
            )

    class FailingCompletions:
        @staticmethod
        def create(**kwargs):
            raise RuntimeError("vllm unavailable")

    class FailingChat:
        completions = FailingCompletions()

    class FailingClient:
        chat = FailingChat()

    monkeypatch.setattr(workbench, "_get_influx_client", lambda: FakeInfluxClient())
    monkeypatch.setattr(workbench, "_get_vllm_client", lambda: FailingClient())

    r = client.post("/insights/api/explain", json={"selected_times": ["2026-01-01T00:00:00Z"]})
    assert_condition(r.status_code == 500)
    assert_condition(r.json()["error"] == "Unable to generate explanation")


def test_insights_explain_handles_influx_client_error(client, monkeypatch):
    def raise_influx_error():
        raise RuntimeError("influx unavailable")

    monkeypatch.setattr(workbench, "_get_influx_client", raise_influx_error)

    r = client.post("/insights/api/explain", json={"selected_times": ["2026-01-01T00:00:00Z"]})
    assert_condition(r.status_code == 500)
    assert_condition(r.json()["error"] == "Unable to process explain request")
