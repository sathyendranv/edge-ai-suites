# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The integrated VLM Reasoning uses the same UI process and proxy."""

from types import SimpleNamespace

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from src import insights
from src import app as ui_app
from src.app import app


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def get_points(self):
        return iter(self.rows)


class FakeInflux:
    def __init__(self, rows_for_query):
        self.rows_for_query = rows_for_query
        self.queries = []
        self.closed = False

    def query(self, sql):
        self.queries.append(sql)
        return FakeResult(self.rows_for_query(sql))

    def close(self):
        self.closed = True


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("REST_API_ROOT_PATH", "/agentic-ui")
    monkeypatch.setenv("AGENTIC_UI_ENABLED", "true")
    with TestClient(app, root_path="/agentic-ui") as test_client:
        yield test_client


def test_workbench_page_and_assets(client):
    page = client.get("/insights-ui/")
    assert page.status_code == 200
    assert "Fusion Analytics Results" in page.text
    assert 'href="/agentic-ui/"' in page.text
    assert 'href="/insights-ui/"' in page.text

    for asset in ["css/style.css", "css/insights.css", "js/insights.js"]:
        response = client.get(f"/insights-ui/static/{asset}")
        assert response.status_code == 200
    assert client.get("/agentic-ui/health").status_code == 200


@respx.mock
def test_dashboard_links_to_insights(client):
    respx.get("http://mock-storage/detections/summary").mock(return_value=httpx.Response(200, json={}))
    respx.get("http://mock-agent/agents/runs").mock(return_value=httpx.Response(200, json=[]))
    respx.get("http://mock-detection/detection/videos").mock(return_value=httpx.Response(200, json={"videos": []}))

    response = client.get("/agentic-ui/")
    assert response.status_code == 200
    assert 'href="/insights-ui/"' in response.text
    assert 'href="/agentic-ui/static/css/style.css"' in response.text


def test_workbench_nav_is_hidden_in_vllm_only_mode(client, monkeypatch):
    monkeypatch.setenv("AGENTIC_UI_ENABLED", "false")
    response = client.get("/insights-ui/")
    assert response.status_code == 200
    assert 'href="/agentic-ui/"' not in response.text
    assert 'href="/insights-ui/"' in response.text


def test_measurements_and_paginated_rows(client, monkeypatch):
    monkeypatch.setenv("FUSION_MEASUREMENT", "custom_fusion")
    influx = FakeInflux(lambda _: [{"time": "t1"}, {"time": "t2"}, {"time": "t3"}])
    monkeypatch.setattr(insights, "get_influx_client", lambda: influx)

    assert client.get("/insights-ui/api/measurements").json() == {"measurements": ["custom_fusion"]}
    response = client.get("/insights-ui/api/data?page=0&page_size=2")
    assert response.status_code == 200
    assert response.json() == {
        "measurement": "custom_fusion",
        "page": 1,
        "page_size": 2,
        "has_more": True,
        "rows": [{"time": "t1"}, {"time": "t2"}],
    }
    assert "FROM custom_fusion" in influx.queries[0]
    assert "LIMIT 3 OFFSET 0" in influx.queries[0]
    assert influx.closed


def test_influx_error_is_reported_without_stopping_ui(client, monkeypatch):
    def unavailable():
        raise ConnectionError("database unavailable")

    monkeypatch.setattr(insights, "get_influx_client", unavailable)
    response = client.get("/insights-ui/api/data")
    assert response.status_code == 500
    assert response.json() == {"error": "Unable to load data", "rows": []}
    assert client.get("/agentic-ui/health").status_code == 200


def test_vllm_readiness(client, monkeypatch):
    class Healthy:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def getcode(self):
            return 200

    monkeypatch.setattr(insights, "urlopen", lambda *args, **kwargs: Healthy())
    assert client.get("/insights-ui/api/vllm/health").json() == {"accessible": True}

    def unavailable(*args, **kwargs):
        raise ConnectionError("vLLM starting")

    monkeypatch.setattr(insights, "urlopen", unavailable)
    response = client.get("/insights-ui/api/vllm/health")
    assert response.status_code == 503
    assert response.json() == {"accessible": False}


def test_explain_fuses_vision_and_sensor_data(client, monkeypatch):
    selected_time = "2026-01-01T12:00:00Z"

    def rows_for_query(sql):
        if "FROM fusion_result" in sql:
            return [{"vision_timestamp": "vision-ts", "timeseries_timestamp": 123}]
        if 'FROM "vision-weld-classification-results"' in sql:
            return [{"frame_id": 8, "img_handle": "frame-8"}]
        if 'FROM "weld-sensor-anomaly-data"' in sql:
            return [{"Primary Weld Current": 80, "Pressure": 2.5}]
        raise AssertionError(sql)

    influx = FakeInflux(rows_for_query)
    monkeypatch.setattr(insights, "get_influx_client", lambda: influx)
    monkeypatch.setattr(insights, "build_image_data_url", lambda _: "data:image/jpeg;base64,AAA=")
    model_requests = []

    def create_completion(**kwargs):
        model_requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="## Weld report"))])

    monkeypatch.setattr(insights.vllm_client.chat.completions, "create", create_completion)
    response = client.post("/insights-ui/api/explain", json={"selected_times": [selected_time]})

    assert response.status_code == 200
    result = response.json()
    assert result["markdown"] == "## Weld report"
    assert result["selected_times"] == [selected_time]
    assert result["resolved_images"][0]["image_load_url"].endswith("/frame-8.jpg")
    assert "Primary Weld Current: 80 A" in result["ts_data"][0]
    assert model_requests[0]["messages"][0] == insights.get_query_prompt()
    assert model_requests[0]["messages"][1]["content"][0]["image_url"]["url"] == "data:image/jpeg;base64,AAA="
    assert influx.closed


def test_explain_rejects_invalid_timestamp(client, monkeypatch):
    influx = FakeInflux(lambda _: [])
    monkeypatch.setattr(insights, "get_influx_client", lambda: influx)
    response = client.post("/insights-ui/api/explain", json={"selected_times": ["bad-timestamp"]})
    assert response.status_code == 400
    assert response.json() == {"error": "Invalid time format: bad-timestamp"}
    assert influx.queries == []
    assert influx.closed


@respx.mock
def test_agentic_results_and_run_still_work(client, monkeypatch):
    respx.get("http://mock-agent/agents/status/run-123").mock(
        return_value=httpx.Response(200, json={"status": "completed"})
    )
    respx.get("http://mock-agent/agents/results/run-123").mock(
        return_value=httpx.Response(200, json={"ticket": {"mode": "fallback", "ticket_id": "t-1", "priority": "LOW", "summary": "Inspect weld", "recommended_action": "Review"}})
    )
    result = client.get("/agentic-ui/results/run-123")
    assert result.status_code == 200
    assert "Inspect weld" in result.text
    assert 'href="/insights-ui/"' in result.text

    published = []
    monkeypatch.setattr(ui_app, "_publish_batch_complete", lambda payload: published.append(payload))
    response = client.post("/agentic-ui/run", data={"time_range": "5m"}, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == f"/agentic-ui/results/{published[0]['run_id']}"
    assert published[0]["end_id"] - published[0]["start_id"] == 5 * 60 * 1_000_000_000