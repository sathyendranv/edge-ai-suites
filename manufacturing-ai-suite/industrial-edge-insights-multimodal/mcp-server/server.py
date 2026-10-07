"""MCP tools for controlling weld detection and explaining fused results."""

import json
import os
import re
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse

PIPELINE_API_URL = os.getenv("PIPELINE_API_URL", "http://dlstreamer-pipeline-server:8080").rstrip("/")
EXPLAIN_API_URL = os.getenv("EXPLAIN_API_URL", "http://multimodal-agentic-ui:5003").rstrip("/")
PIPELINE_REQUEST_PATH = (
    Path(__file__).resolve().parent.parent
    / "configs/dlstreamer-pipeline-server/pipeline-request-cpu.json"
)

mcp = FastMCP("Weld defect detection", host="0.0.0.0", port=8000, stateless_http=True)


class StartRequest(BaseModel):
    device: Literal["CPU", "GPU", "NPU"] = "CPU"


class StopRequest(BaseModel):
    pipeline_id: str


class ExplainRequest(BaseModel):
    selected_time: str


async def _request(method: str, url: str, **kwargs: Any) -> httpx.Response:
    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=5), trust_env=False) as client:
        response = await client.request(method, url, **kwargs)
        response.raise_for_status()
        return response


@mcp.tool()
async def start_pipeline(device: Literal["CPU", "GPU", "NPU"] = "CPU") -> dict[str, Any]:
    """Start weld defect classification on the selected inference device."""
    payload = json.loads(PIPELINE_REQUEST_PATH.read_text(encoding="utf-8"))
    payload["parameters"]["classification-properties"]["device"] = device
    response = await _request(
        "POST",
        f"{PIPELINE_API_URL}/pipelines/user_defined_pipelines/weld_defect_classification",
        json=payload,
    )
    return response.json()


@mcp.tool()
async def stop_pipeline(pipeline_id: str) -> dict[str, str]:
    """Stop one pipeline by ID, leaving other pipelines running."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", pipeline_id):
        raise ValueError("pipeline_id must be an alphanumeric pipeline ID")
    await _request("DELETE", f"{PIPELINE_API_URL}/pipelines/{pipeline_id}")
    return {"stopped_pipeline_id": pipeline_id}


@mcp.tool()
async def explain(selected_time: str) -> dict[str, Any]:
    """Explain one fused weld detection by its ISO-8601 timestamp."""
    response = await _request(
        "POST",
        f"{EXPLAIN_API_URL}/insights-ui/api/explain",
        json={"selected_times": [selected_time]},
    )
    return response.json()


async def _http_tool(
    request: Request,
    schema: type[BaseModel],
    action: Callable[..., Awaitable[dict[str, Any]]],
) -> JSONResponse:
    try:
        payload = schema.model_validate(await request.json())
        return JSONResponse(await action(**payload.model_dump()))
    except (ValueError, ValidationError) as error:
        return JSONResponse({"error": str(error)}, status_code=400)
    except httpx.HTTPStatusError as error:
        return JSONResponse(
            {"error": f"Upstream returned HTTP {error.response.status_code}"},
            status_code=error.response.status_code,
        )
    except httpx.RequestError:
        return JSONResponse({"error": "Upstream service unavailable"}, status_code=502)


@mcp.custom_route("/start_pipeline", methods=["POST"])
async def http_start_pipeline(request: Request) -> JSONResponse:
    return await _http_tool(request, StartRequest, start_pipeline)


@mcp.custom_route("/stop_pipeline", methods=["POST"])
async def http_stop_pipeline(request: Request) -> JSONResponse:
    return await _http_tool(request, StopRequest, stop_pipeline)


@mcp.custom_route("/explain", methods=["POST"])
async def http_explain(request: Request) -> JSONResponse:
    return await _http_tool(request, ExplainRequest, explain)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")