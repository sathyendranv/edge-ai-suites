# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Set test backends before either UI module imports the FastAPI application."""

import os

os.environ["MQTT_DISABLED"] = "true"
os.environ["AGENT_SERVICE_URL"] = "http://mock-agent"
os.environ["DETECTION_SERVICE_URL"] = "http://mock-detection"
os.environ["STORAGE_SERVICE_URL"] = "http://mock-storage"
os.environ["USE_CASE_ID"] = "test-case"