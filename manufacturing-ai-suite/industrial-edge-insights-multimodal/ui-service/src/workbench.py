# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import datetime
import json
import logging
import mimetypes
import os
import re
from typing import Any
from urllib.request import urlopen

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from influxdb import InfluxDBClient
from openai import OpenAI
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

_src_dir = os.path.dirname(__file__)
_templates = Jinja2Templates(directory=os.path.join(_src_dir, "templates"))

router = APIRouter(prefix="/insights")
_MEASUREMENT_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_MAX_SELECTED_TIMES = 5


class ExplainRequest(BaseModel):
    selected_times: list[str] = Field(default_factory=list)


def _workbench_path(request: Request) -> str:
    root_path = request.scope.get("root_path", "")
    return f"{root_path}{request.app.url_path_for('workbench_page')}"


def _dashboard_path(request: Request) -> str:
    route_name = "dashboard_page" if os.getenv("UI_DEFAULT_PAGE", "dashboard").lower() == "insights" else "root_page"
    root_path = request.scope.get("root_path", "")
    return f"{root_path}{request.app.url_path_for(route_name)}"


def _show_agentic_tabs() -> bool:
    return os.getenv("UI_ENABLE_AGENTIC_TABS", "true").lower() == "true"


def _get_vllm_client() -> OpenAI:
    return OpenAI(
        base_url=(
            f"http://{os.getenv('VLLM_HOST', 'vllm-server')}:{os.getenv('VLLM_PORT', '8000')}/v1"
        ),
        api_key="EMPTY",
    )


def _get_seaweed_public_image_base_path() -> str:
    return (
        f"{os.getenv('OBJECT_STORE_URL', 'http://seaweedfs-filer:8888')}/buckets/"
        f"{os.getenv('BUCKET_NAME', 'dlstreamer-pipeline-results/weld-defect-classification')}"
    ).rstrip("/")


def _build_image_url(img_handle: str) -> str:
    return f"{_get_seaweed_public_image_base_path()}/{img_handle}.jpg"


def _build_image_data_url(image_url: str) -> str | None:
    try:
        with urlopen(image_url, timeout=10) as response:  # nosec B310
            image_bytes = response.read()
            header_mime = response.headers.get_content_type()

        mime_type = header_mime if header_mime and header_mime != "application/octet-stream" else None
        if not mime_type:
            guessed_mime, _ = mimetypes.guess_type(image_url)
            mime_type = guessed_mime or "image/jpeg"

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{b64_image}"
    except Exception as exc:  # noqa: BLE001
        log.warning("Unable to build data URL for image=%s error=%s", image_url, exc)
        return None


def _get_query_prompt() -> dict[str, Any]:
    prompt_path = os.path.join(_src_dir, "system_prompt.json")
    with open(prompt_path, encoding="utf-8") as prompt_file:
        return json.load(prompt_file)


def _get_fusion_measurement_name() -> str:
    measurement = os.getenv("FUSION_MEASUREMENT", "fusion_result")
    if not _MEASUREMENT_RE.fullmatch(measurement):
        log.warning("Invalid FUSION_MEASUREMENT=%s; falling back to fusion_result", measurement)
        return "fusion_result"
    return measurement


def _get_vllm_health_url() -> str:
    host = os.getenv("VLLM_HOST", "vllm-server")
    port = os.getenv("VLLM_PORT", "8000")
    return f"http://{host}:{port}/docs"


def _get_vllm_max_tokens() -> int:
    return int(os.getenv("VLLM_MAX_TOKENS", os.getenv("VLLM_CLIENT_TOKEN", "2048")))


def _get_influx_client() -> InfluxDBClient:
    host = os.getenv("INFLUX_HOST", "localhost")
    port = int(os.getenv("INFLUX_PORT", "8086"))
    username = os.getenv("INFLUX_USER", "admin")
    password = os.getenv("INFLUX_PASSWORD", "admin")
    database = os.getenv("INFLUX_DB", "datain")

    return InfluxDBClient(
        host=host,
        port=port,
        username=username,
        password=password,
        database=database,
        timeout=10,
    )


def _normalize_iso_timestamp(value: str) -> str:
    parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    else:
        parsed = parsed.astimezone(datetime.timezone.utc)
    return parsed.isoformat().replace("+00:00", "Z")


def _normalize_numeric_timestamp(value: Any) -> int:
    return int(value)


def _fetch_rows(
    client: InfluxDBClient,
    page: int,
    page_size: int,
) -> tuple[list[dict[str, Any]], bool]:
    offset = (page - 1) * page_size
    no_result_re = r"/^(No_Weld|No Weld|No_Label|No Label|No label)$/"
    measurement = _get_fusion_measurement_name()
    query = (
        f"SELECT time, timeseries_classification, vision_classification, fused_decision FROM {measurement} "
        f"WHERE vision_classification !~ {no_result_re} "
        f"AND timeseries_classification !~ {no_result_re} "
        f"ORDER BY time DESC LIMIT {page_size + 1} OFFSET {offset}"
    )  # nosec B608

    result = client.query(query)
    points = list(result.get_points())
    has_more = len(points) > page_size
    rows = points[:page_size]

    return rows, has_more


@router.get("", response_class=HTMLResponse, name="workbench_page")
def workbench_page(request: Request):
    return _templates.TemplateResponse(
        request=request,
        name="workbench.html",
        context={
            "workbench_root": _workbench_path(request),
            "dashboard_href": _dashboard_path(request),
            "insights_href": _workbench_path(request),
            "show_agentic_tabs": _show_agentic_tabs(),
        },
    )


@router.get("/api/measurements")
def workbench_measurements() -> dict[str, list[str]]:
    measurement = _get_fusion_measurement_name()
    return {"measurements": [measurement]}


@router.get("/api/data")
def workbench_data(page: int = 1, page_size: int = 10):
    measurement = _get_fusion_measurement_name()
    safe_page = max(page, 1)
    safe_page_size = max(min(page_size, 200), 1)

    try:
        client = _get_influx_client()
        rows, has_more = _fetch_rows(client, safe_page, safe_page_size)
        return {
            "measurement": measurement,
            "page": safe_page,
            "page_size": safe_page_size,
            "has_more": has_more,
            "rows": rows,
        }
    except Exception:  # noqa: BLE001
        log.exception("Failed to fetch fusion rows for measurement=%s", measurement)
        return JSONResponse({"error": "Unable to load data", "rows": []}, status_code=500)


@router.get("/api/vllm/health")
def workbench_vllm_health():
    health_url = _get_vllm_health_url()
    log.info("Checking vLLM endpoint: %s", health_url)

    try:
        with urlopen(health_url, timeout=5) as response:  # nosec B310
            status_code = response.getcode()

        accessible = 200 <= status_code < 400
        return {"accessible": accessible}
    except Exception as exc:  # noqa: BLE001
        log.error("Unable to access vLLM endpoint=%s error=%s", health_url, exc)
        return JSONResponse({"accessible": False}, status_code=503)


@router.post("/api/explain")
async def workbench_explain(payload: ExplainRequest):
    selected_times = payload.selected_times
    if len(selected_times) > _MAX_SELECTED_TIMES:
        return JSONResponse(
            {"error": f"Select at most {_MAX_SELECTED_TIMES} timestamps per request"},
            status_code=400,
        )

    log.info("Explain request received with %d selected time(s)", len(selected_times))
    ts_data: list[str] = []
    resolved_images: list[dict[str, Any]] = []
    message: dict[str, Any] = {"role": "user", "content": []}
    client = _get_influx_client()
    measurement = _get_fusion_measurement_name()

    for time_str in selected_times:
        try:
            normalized_time = _normalize_iso_timestamp(time_str)
            query = f"SELECT * FROM {measurement} WHERE time = '{normalized_time}'"  # nosec B608
            result = client.query(query)
            points = list(result.get_points())
            if not points:
                log.warning("No fusion_result row found for time=%s", time_str)
                continue

            row = points[0]
            vision_timestamp = row.get("vision_timestamp")
            if not vision_timestamp:
                log.warning("No vision_timestamp found in fusion row for time=%s", time_str)
                continue
            normalized_vision_timestamp = _normalize_iso_timestamp(str(vision_timestamp))

            query_vision = (
                'SELECT * FROM "vision-weld-classification-results" '
                f"WHERE search_time = '{normalized_vision_timestamp}'"
            )  # nosec B608
            result_vision = client.query(query_vision)
            points_vision = list(result_vision.get_points())

            img_handle = None
            image_url = None
            image_data_url = None
            frame_id = None
            if points_vision:
                frame_id = points_vision[0].get("frame_id")
                img_handle = points_vision[0].get("img_handle")
                image_url = _build_image_url(str(img_handle)) if img_handle else None
                image_data_url = _build_image_data_url(image_url) if image_url else None
                resolved_images.append(
                    {
                        "selected_time": time_str,
                        "frame_id": frame_id,
                        "img_handle": img_handle,
                        "image_url": image_url,
                        "image_load_url": (
                            f"/image-store/buckets/{os.getenv('BUCKET_NAME', 'dlstreamer-pipeline-results/weld-defect-classification')}/{img_handle}.jpg"
                            if img_handle
                            else None
                        ),
                    }
                )

            normalized_sensor_timestamp = _normalize_numeric_timestamp(row.get("timeseries_timestamp"))
            query_sensor = (
                f'SELECT * FROM "weld-sensor-anomaly-data" '
                f"WHERE time = {normalized_sensor_timestamp}"
            )  # nosec B608
            result_sensor = client.query(query_sensor)
            points_sensor = list(result_sensor.get_points())
            if not points_sensor:
                log.warning("No sensor data found for time=%s", time_str)
                continue

            sensor_row = points_sensor[0]
            sensor_text = f"""
                Sensor Data:
                    • Primary Weld Current: {sensor_row.get('Primary Weld Current', 'N/A')} A
                    • Secondary Weld Voltage: {sensor_row.get('Secondary Weld Voltage', 'N/A')} V
                    • Pressure: {sensor_row.get('Pressure', 'N/A')} bar
                    • CO2 Weld Flow: {sensor_row.get('CO2 Weld Flow', 'N/A')} L/min
                    • Feed: {sensor_row.get('Feed', 'N/A')} mm/min
                    • Wire Consumed: {sensor_row.get('Wire Consumed', 'N/A')} mm
                """

            if image_data_url:
                message["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    }
                )
            message["content"].append(
                {
                    "type": "text",
                    "text": """
                Given this weld image and the sensor telemetry, produce a structured
                weld quality report covering defect classification, root cause, and remediation steps.
                """
                    + sensor_text,
                }
            )
            ts_data.append(sensor_text)
        except ValueError:
            return JSONResponse({"error": f"Invalid time format: {time_str}"}, status_code=400)
        except Exception:  # noqa: BLE001
            log.exception("Explain processing failed for time=%s", time_str)
            return JSONResponse({"error": "Unable to process explain request"}, status_code=500)

    if not message["content"]:
        return JSONResponse({"error": "No valid data found for the selected timestamp(s)"}, status_code=400)

    try:
        response = _get_vllm_client().chat.completions.create(
            model=os.getenv("VLLM_ADAPTER_NAME", "qwen3.5-2b-adapter"),
            messages=[_get_query_prompt(), message],
            max_tokens=_get_vllm_max_tokens(),
            temperature=float(os.getenv("VLLM_CLIENT_TEMPERATURE", "1.5")),
            extra_body={
                "min_p": float(os.getenv("VLLM_CLIENT_MIN_P", "0.1")),
            },
        )
    except Exception:  # noqa: BLE001
        log.exception("Explain request failed during vLLM completion")
        return JSONResponse({"error": "Unable to generate explanation"}, status_code=500)

    markdown = ""
    if response.choices:
        markdown = response.choices[0].message.content or ""

    return {
        "title": "AI Assistant Output",
        "markdown": markdown,
        "selected_times": selected_times,
        "resolved_images": resolved_images,
        "ts_data": ts_data,
    }
