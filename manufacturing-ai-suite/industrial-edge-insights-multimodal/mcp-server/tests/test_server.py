"""Behavioral tests for the standalone weld MCP service."""

import json
import unittest
from unittest.mock import AsyncMock, patch

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
        self.assertEqual({"start_pipeline", "stop_pipeline", "explain"}, {tool.name for tool in tools})

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