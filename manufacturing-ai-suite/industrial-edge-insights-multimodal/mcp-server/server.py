"""MCP tools for controlling weld detection and explaining fused results."""

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal
from urllib.parse import urlsplit

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse

PIPELINE_API_URL = os.getenv("PIPELINE_API_URL", "http://dlstreamer-pipeline-server:8080").rstrip("/")
EXPLAIN_API_URL = os.getenv("EXPLAIN_API_URL", "http://multimodal-agentic-ui:5003").rstrip("/")
AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "http://apm-agent:5002").rstrip("/")
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


async def _request(method: str, url: str, *, allow_redirect: bool = False, **kwargs: Any) -> httpx.Response:
    async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=5), trust_env=False) as client:
        response = await client.request(method, url, **kwargs)
        if not (allow_redirect and response.status_code == 303):
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


@mcp.tool()
async def list_insights_data(page: int = 1, page_size: int = 10) -> dict[str, Any]:
    """List paginated weld fusion results from the insights workbench."""
    response = await _request(
        "GET",
        f"{EXPLAIN_API_URL}/insights-ui/api/data",
        params={"page": page, "page_size": page_size},
    )
    return response.json()


@mcp.tool()
async def run_agent(time_range: Literal["30s", "1m", "5m", "10m", "30m"] = "30s") -> dict[str, str]:
    """Start agentic weld analysis for a selected time range, returning the run ID."""
    response = await _request(
        "POST", f"{EXPLAIN_API_URL}/run", data={"time_range": time_range},
        follow_redirects=False, allow_redirect=True,
    )
    results_path = urlsplit(response.headers.get("location", "")).path
    if response.status_code != 303 or "/results/" not in results_path:
        raise ValueError("Agentic UI did not return a run results redirect")
    run_id = results_path.rsplit("/results/", 1)[1]
    try:
        uuid.UUID(run_id)
    except ValueError as error:
        raise ValueError("Agentic UI returned an invalid run ID") from error
    return {"run_id": run_id, "results_path": results_path}


@mcp.tool()
async def get_run_results(run_id: str) -> dict[str, Any]:
    """Get the agent status or completed result shown on the run results page."""
    try:
        uuid.UUID(run_id)
    except ValueError as error:
        raise ValueError("run_id must be a UUID") from error

    status_response = await _request("GET", f"{AGENT_SERVICE_URL}/agents/status/{run_id}")
    status = status_response.json()
    if status.get("status") == "completed":
        result_response = await _request("GET", f"{AGENT_SERVICE_URL}/agents/results/{run_id}")
        return {"run_id": run_id, "phase": "completed", "result": result_response.json()}
    return {"run_id": run_id, "phase": "reasoning", "result": status}


@mcp.tool()
async def describe() -> dict[str, Any]:
    """Describe the weld-analysis MCP server, registered tools, and operational scope."""
    tools = await mcp.list_tools()
    return {
        "name": "Weld defect detection",
        "purpose": "Pipeline control, fusion data access, and agent-assisted weld quality analysis.",
        "tools": [
            {"name": tool.name, "description": tool.description or "", "input_schema": tool.inputSchema}
            for tool in tools
        ],
        "operational_guidance": (
            "Pipeline start/stop commands affect live processing. Agent and model-generated "
            "analysis is advisory; review results before making operational or safety decisions."
        ),
    }


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