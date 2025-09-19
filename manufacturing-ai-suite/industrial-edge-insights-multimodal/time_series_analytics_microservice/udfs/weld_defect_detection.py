#
# Apache v2 license
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

""" Custom user defined function for anomaly detection for weld_defect_detection. """

import os
import logging
import pickle
import time
import math
import warnings
from collections import deque
from kapacitor.udf.agent import Agent, Handler
from kapacitor.udf import udf_pb2
import numpy as np
import requests
from sklearnex import patch_sklearn, config_context
patch_sklearn()
import torch
from torch.utils.data import DataLoader, Dataset
from my_TimesNet import Model as my_TimesNet  # Import custom TimesNet with Inception_Block_V2

warnings.filterwarnings(
    "ignore",
    message=".*Threading.*parallel backend is not supported by Extension for Scikit-learn.*"
)


log_level = os.getenv('KAPACITOR_LOGGING_LEVEL', 'INFO').upper()
enable_benchmarking = os.getenv('ENABLE_BENCHMARKING', 'false').upper() == 'TRUE'
total_no_pts = int(os.getenv('BENCHMARK_TOTAL_PTS', "0"))
logging_level = getattr(logging, log_level, logging.INFO)

# Configure logging
logging.basicConfig(
    level=logging_level,  # Set the log level to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log format
)

logger = logging.getLogger()

# Anomaly detection on the windturbine speed and generated power data
class AnomalyDetectorHandler(Handler):
    """ Handler for the anomaly detection UDF. It processes incoming points
    and detects anomalies based on the wind speed and generated power data.
    """
    def __init__(self, agent):
        self._agent = agent
        # Need to enable after model training
        model_name = (os.path.basename(__file__)).replace('.py', '.pth')
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "../models/" + model_name)
        model_path = os.path.abspath(model_path)
        # self.rf = load_model(model_path)

        config = {
            "activation": "gelu",
            "c_out": 6,
            "d_ff": 128,
            "d_layers": 1,
            "d_model": 128,
            "dec_in": 6,
            "dropout": 0.0,
            "e_layers": 3,
            "embed": "timeF",
            "enc_in": 6,
            "epochs": 2,
            "factor": 1,
            "freq": "h",
            "moving_avg": 10,
            "n_heads": 4,
            "num_kernels": 4,
            "output_attention": False,
            "pred_len": 0,
            "seq_len": 384,
            "top_k": 3
        }

        self.model = my_TimesNet(config)

        # device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        self.model.load_state_dict(torch.load(model_path, map_location=device))

        self.points_received = {}
        global total_no_pts
        self.max_points = int(total_no_pts)

    def info(self):
        """ Return the InfoResponse. Describing the properties of this Handler
        """
        response = udf_pb2.Response()
        response.info.wants = udf_pb2.STREAM
        response.info.provides = udf_pb2.STREAM
        return response

    def init(self, init_req):
        """ Initialize the Handler with the provided options.
        """
        response = udf_pb2.Response()
        response.init.success = True
        return response

    def snapshot(self):
        """ Create a snapshot of the running state of the process.
        """
        response = udf_pb2.Response()
        response.snapshot.snapshot = b''
        return response

    def restore(self, restore_req):
        """ Restore a previous snapshot.
        """
        response = udf_pb2.Response()
        response.restore.success = False
        response.restore.error = 'not implemented'
        return response

    def begin_batch(self, begin_req):
        """ A batch has begun.
        """
        raise Exception("not supported")

    def point(self, point):
        """ A point has arrived.
        """
        server = None
        for point_tag in point.tags:
            if point_tag.key == "source":
                server = point_tag.value
                break
        global enable_benchmarking
        if enable_benchmarking:
            if server not in self.points_received:
                self.points_received[server] = 0
            if self.points_received[server] >= self.max_points:
                return
            self.points_received[server] += 1

        logger.info("Processing point %s %s for source %s", point.time, time.time(), server)

        response = udf_pb2.Response()
        if not any(kv.key == "anomaly_status" for kv in point.fieldsDouble):
            point.fieldsDouble.add(key = "anomaly_status", value = 0.0)
            
        response.point.CopyFrom(point)
        self._agent.write_response(response, True)

    def end_batch(self, end_req):
        """ The batch is complete.
        """
        raise Exception("not supported")


if __name__ == '__main__':
    # Create an agent
    agent = Agent()

    # Create a handler and pass it an agent so it can write points
    h = AnomalyDetectorHandler(agent)

    # Set the handler on the agent
    agent.handler = h

    # Anything printed to STDERR from a UDF process gets captured
    # into the Kapacitor logs.
    agent.start()
    agent.wait()
