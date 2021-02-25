from artificienlib import syftfunctions
from artificienlib import constants

import syft as sy
from syft.serde import protobuf
from syft_proto.execution.v1.plan_pb2 import Plan as PlanPB
from syft_proto.execution.v1.state_pb2 import State as StatePB
from syft.grid.clients.model_centric_fl_client import ModelCentricFLClient
from syft.execution.state import State
from syft.execution.placeholder import PlaceHolder
from syft.execution.translation import TranslationTarget

import torch as th
from torch import nn

import os
from websocket import create_connection
import websockets
import json
import requests

sy.make_hook(globals())
hook.local_worker.framework = None # force protobuf serialization for tensors
th.random.manual_seed(1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 392)
        self.fc2 = nn.Linear(392, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

model = Net()

def test_mse():
    batch_size = 2
    logits = th.tensor((), dtype=th.float64)
    logits.new_ones((2, 3))
    targets = th.tensor((), dtype=th.float64)
    targets.new_zeros((2,3))
    assert syftfunctions.mse_with_logits(logits, targets, batch_size) == (logits - targets).sum() / batch_size

def test_2_node():
    
    X = th.randn(3, 28 * 28)
    y = nn.functional.one_hot(th.tensor([1, 2, 3]), 10)
    
    model_params, training_plan = syftfunctions.def_training_plan(model, X, y)

    avg_plan = syftfunctions.def_avg_plan(model_params)

    grid = syftfunctions.artificien_connect()

    syftfunctions.send_model(name="lib_test", version="0.1.0", batch_size=2, learning_rate=0.2, max_updates=10, model_params=model_params, grid=grid, training_plan=training_plan, avg_plan=avg_plan)
    
    assert True




    