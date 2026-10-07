"""Behavioral tests for the standalone weld MCP service."""

import json
import unittest
from unittest.mock import AsyncMock, call, patch

import httpx

import server


class WeldMCPServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.mcp.streamable_http_app()),
            base_url="http://localhost:8000",
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_mcp_tools_are_registered(self) -> None:
        tools = await server.mcp.list_tools()
        self.assertEqual(
            {"start_pipeline", "stop_pipeline", "explain", "run_agent", "get_run_results"},
            {tool.name for tool in tools},
        )
        run_tool = next(tool for tool in tools if tool.name == "run_agent")
        time_range = run_tool.inputSchema["properties"]["time_range"]
        self.assertEqual(["30s", "1m", "5m", "10m", "30m"], time_range["enum"])
        self.assertEqual("30s", time_range["default"])

    async def test_start_pipeline_uses_checked_in_payload(self) -> None:
        original = json.loads(server.PIPELINE_REQUEST_PATH.read_text(encoding="utf-8"))
        for device in ("CPU", "GPU", "NPU"):
            with self.subTest(device=device), patch.object(server, "_request", new_callable=AsyncMock) as upstream:
                upstream.return_value = httpx.Response(200, json={"id": "pipeline-1"})
                self.assertEqual({"id": "pipeline-1"}, await server.start_pipeline(device))
                sent = upstream.await_args
                self.assertEqual("POST", sent.args[0])
                self.assertEqual(
                    "http://dlstreamer-pipeline-server:8080/pipelines/"
                    "user_defined_pipelines/weld_defect_classification",
                    sent.args[1],
                )
                expected = json.loads(json.dumps(original))
                expected["parameters"]["classification-properties"]["device"] = device
                self.assertEqual(expected, sent.kwargs["json"])
        self.assertEqual(original, json.loads(server.PIPELINE_REQUEST_PATH.read_text(encoding="utf-8")))

    async def test_stop_targets_only_the_requested_id(self) -> None:
        with patch.object(server, "_request", new_callable=AsyncMock) as upstream:
            self.assertEqual(
                {"stopped_pipeline_id": "pipeline-1"},
                await server.stop_pipeline("pipeline-1"),
            )
            upstream.assert_awaited_once_with(
                "DELETE", "http://dlstreamer-pipeline-server:8080/pipelines/pipeline-1"
            )
            with self.assertRaises(ValueError):
                await server.stop_pipeline("../other")
            upstream.assert_awaited_once()

    async def test_explain_forwards_single_timestamp(self) -> None:
        with patch.object(server, "_request", new_callable=AsyncMock) as upstream:
            upstream.return_value = httpx.Response(200, json={"markdown": "Weld report"})
            self.assertEqual({"markdown": "Weld report"}, await server.explain("2026-10-01T12:00:00Z"))
            upstream.assert_awaited_once_with(
                "POST",
                "http://multimodal-agentic-ui:5003/insights-ui/api/explain",
                json={"selected_times": ["2026-10-01T12:00:00Z"]},
            )

    async def test_run_agent_returns_id_from_ui_redirect(self) -> None:
        run_id = "123e4567-e89b-12d3-a456-426614174000"
        with patch.object(server, "_request", new_callable=AsyncMock) as upstream:
            upstream.return_value = httpx.Response(
                303, headers={"location": f"/agentic-ui/results/{run_id}"}
            )
            self.assertEqual(
                {"run_id": run_id, "results_path": f"/agentic-ui/results/{run_id}"},
                await server.run_agent("5m"),
            )
            upstream.assert_awaited_once_with(
                "POST", "http://multimodal-agentic-ui:5003/run",
                data={"time_range": "5m"}, follow_redirects=False, allow_redirect=True,
            )
            upstream.reset_mock()
            self.assertEqual(run_id, (await server.run_agent())["run_id"])
            upstream.assert_awaited_once_with(
                "POST", "http://multimodal-agentic-ui:5003/run",
                data={"time_range": "30s"}, follow_redirects=False, allow_redirect=True,
            )
            upstream.return_value = httpx.Response(200, json={"status": "error"})
            with self.assertRaises(ValueError):
                await server.run_agent()
            upstream.return_value = httpx.Response(303, headers={"location": "/other"})
            with self.assertRaises(ValueError):
                await server.run_agent()

    async def test_run_agent_handles_http_redirect_without_masking_upstream_errors(self) -> None:
        run_id = "9bf093d6-9e2d-4527-a754-a060bda632a6"

        def redirect(request: httpx.Request) -> httpx.Response:
            self.assertEqual("POST", request.method)
            self.assertEqual(b"time_range=30s", request.content)
            return httpx.Response(303, headers={"location": f"/agentic-ui/results/{run_id}"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(redirect))
        with patch.object(server.httpx, "AsyncClient", return_value=client):
            self.assertEqual(run_id, (await server.run_agent())["run_id"])

        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(502)))
        with patch.object(server.httpx, "AsyncClient", return_value=client):
            with self.assertRaises(httpx.HTTPStatusError):
                await server.run_agent()

    async def test_get_run_results_matches_agent_status_and_completed_result(self) -> None:
        run_id = "123e4567-e89b-12d3-a456-426614174000"
        status_url = f"http://apm-agent:5002/agents/status/{run_id}"
        results_url = f"http://apm-agent:5002/agents/results/{run_id}"
        with patch.object(server, "_request", new_callable=AsyncMock) as upstream:
            upstream.side_effect = [
                httpx.Response(200, json={"status": "running"}),
                httpx.Response(200, json={"status": "completed"}),
                httpx.Response(200, json={"ticket": "Inspect weld"}),
            ]
            self.assertEqual(
                {"run_id": run_id, "phase": "reasoning", "result": {"status": "running"}},
                await server.get_run_results(run_id),
            )
            self.assertEqual(
                {"run_id": run_id, "phase": "completed", "result": {"ticket": "Inspect weld"}},
                await server.get_run_results(run_id),
            )
            self.assertEqual(
                [call("GET", status_url), call("GET", status_url), call("GET", results_url)],
                upstream.await_args_list,
            )

    async def test_get_run_results_rejects_invalid_id(self) -> None:
        with patch.object(server, "_request", new_callable=AsyncMock) as upstream:
            with self.assertRaises(ValueError):
                await server.get_run_results("../other")
            upstream.assert_not_awaited()

    async def test_http_paths_share_tool_actions(self) -> None:
        with patch.object(server, "_request", new_callable=AsyncMock) as upstream:
            upstream.return_value = httpx.Response(200, json={"id": "pipeline-1"})
            response = await self.client.post("/start_pipeline", json={"device": "GPU"})
            self.assertEqual(200, response.status_code)
            self.assertEqual({"id": "pipeline-1"}, response.json())
            self.assertEqual(
                "GPU", upstream.await_args.kwargs["json"]["parameters"]["classification-properties"]["device"]
            )

            response = await self.client.post("/stop_pipeline", json={"pipeline_id": "pipeline-1"})
            self.assertEqual(200, response.status_code)
            self.assertEqual({"stopped_pipeline_id": "pipeline-1"}, response.json())

            upstream.return_value = httpx.Response(200, json={"markdown": "Weld report"})
            response = await self.client.post("/explain", json={"selected_time": "2026-10-01T12:00:00Z"})
            self.assertEqual(200, response.status_code)
            self.assertEqual({"markdown": "Weld report"}, response.json())

    async def test_http_rejects_bad_input_and_reports_upstream_failures(self) -> None:
        with patch.object(server, "_request", new_callable=AsyncMock) as upstream:
            for path, payload in (
                ("/start_pipeline", {"device": "FPGA"}),
                ("/stop_pipeline", {"pipeline_id": "../other"}),
                ("/explain", {}),
            ):
                with self.subTest(path=path):
                    response = await self.client.post(path, json=payload)
                    self.assertEqual(400, response.status_code)
            upstream.assert_not_awaited()

            request = httpx.Request("POST", "http://multimodal-agentic-ui:5003/insights-ui/api/explain")
            upstream.side_effect = httpx.HTTPStatusError(
                "Not found", request=request, response=httpx.Response(404, request=request)
            )
            response = await self.client.post("/explain", json={"selected_time": "2026-10-01T12:00:00Z"})
            self.assertEqual(404, response.status_code)

            upstream.side_effect = httpx.ConnectError("Connection refused", request=request)
            response = await self.client.post("/explain", json={"selected_time": "2026-10-01T12:00:00Z"})
            self.assertEqual(502, response.status_code)