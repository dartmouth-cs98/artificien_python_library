from .constants import masterNode, region_name, userPoolId, clientId

import syft as sy
from syft.serde import protobuf
from syft_proto.execution.v1.plan_pb2 import Plan as PlanPB
from syft_proto.execution.v1.state_pb2 import State as StatePB
from syft.grid.clients.model_centric_fl_client import ModelCentricFLClient
from syft.execution.state import State
from syft.execution.placeholder import PlaceHolder
from syft.execution.translation import TranslationTarget
from syft.grid.clients.model_centric_fl_client import ModelCentricFLClient
from datetime import date

import torch as th

import os
from websocket import create_connection
import json
from requests.structures import CaseInsensitiveDict
import time
import requests
import boto3
from warrant import Cognito

try:
  ecs_client = boto3.client('ecs', region_name=region_name)
except BaseException as exe:
    print(exe)

dynamodb = boto3.resource('dynamodb', region_name=region_name)

os.environ['AWS_DEFAULT_REGION'] = region_name

sy.make_hook(globals())
hook.local_worker.framework = None  # force protobuf serialization for tensors
th.random.manual_seed(1)

def get_my_purchased_datasets(password):
    """ Returns the datasets the user has purchased access to
        Args:
            * username: current user's username (duh)
    """
    user_id = os.environ['JUPYTERHUB_USER']
    try:
        u = Cognito(userPoolId, clientId, username=user_id)
        u.authenticate(password=password)
    except:
        return {"Error": "Failed to authenticate User"}

    accessId = u.access_token
    # PyGrid masterNode address from constants.py
    masterNodeAddy = 'http://' + masterNode + '/get_datasets'

    # create header with cognito access token and json content specifier
    headers = CaseInsensitiveDict()
    headers["Authorization"] = "Bearer " + accessId
    headers["Content-Type"] = "application/json"
    node = {"user_id": user_id}
    resp = requests.post(masterNodeAddy, headers=headers, json=node)
    resp = resp.json()
    print(resp)
    return

# Define some standard loss functions
def mse_with_logits(logits, targets, batch_size):
    """ Calculates mse
        Args:
            * logits: (NxC) outputs of dense layer
            * targets: (NxC) labels
            * batch_size: value of N, temporarily required because Plan cannot trace .shape
    """
    return (logits - targets).sum() / batch_size


def softmax_cross_entropy_with_logits(logits, targets, batch_size):
    """ Calculates softmax entropy
        Args:
            * logits: (NxC) outputs of dense layer
            * targets: (NxC) one-hot encoded labels
            * batch_size: value of N, temporarily required because Plan cannot trace .shape
    """
    # numstable logsoftmax
    norm_logits = logits - logits.max()
    log_probs = norm_logits - norm_logits.exp().sum(dim=1, keepdim=True).log()
    # NLL, reduction = mean
    return -(targets * log_probs).sum() / batch_size

def absolute_error(logits, targets, batch_size):
    """ Calculates absolute error
        Args:
            * logits: (NxC) outputs of dense layer
            * targets: (NxC) one-hot encoded labels
            * batch_size: value of N, temporarily required because Plan cannot trace .shape
    """
    return abs(logits - targets) / batch_size

def binary_cross_entropy(logits, targets, batch_size):
    """ Calculates binary cross entropy lose
        Args:
            * logits: (NxC) outputs of dense layer
            * targets: (NxC) one-hot encoded labels
            * batch_size: value of N, temporarily required because Plan cannot trace .shape
    """
    sum_score = (logits * targets.sum(dim=1, keepdim=True).log()).sum()
    mean_sum_score = sum_score / batch_size
    return -mean_sum_score

# Define standard optimizers
def naive_sgd(param, **kwargs):
    """ Naive Standard Gradient Descent"""
    return param - kwargs['lr'] * param.grad

# Standard function will set tensors as model parameters
def set_model_params(module, params_list, start_param_idx=0):
    """ Set params list into model recursively """
    param_idx = start_param_idx

    for name, param in module._parameters.items():
        module._parameters[name] = params_list[param_idx]
        param_idx += 1

    for name, child in module._modules.items():
        if child is not None:
            param_idx = set_model_params(child, params_list, param_idx)

    return param_idx


def def_training_plan(model, X, y, plan_dict=None):

    """
    :param model: A model built in pytorch
    :param X: Input data
    :param y: Labels
    :param plan_dict: A dictionary representing attributes of the training plan. Values are set to defaults if not set.
    :return: Model parameters and a training plan to be used with pysyft functions
    """

    if plan_dict is None:
        plan_dict = {}
    if 'loss' in plan_dict:
        loss_func = plan_dict['loss']
    else:
        loss_func = softmax_cross_entropy_with_logits
    
    if 'optimizer' in plan_dict:
        optim_func = plan_dict['optimizer']
    else:
        optim_func = naive_sgd
        
    if 'training_plan' in plan_dict:
        @sy.func2plan()
        def training_plan(X, y, batch_size, lr, model_params):
            plan_dict['training_plan'](X, y, batch_size, lr, model_params)
    else:
        @sy.func2plan()
        def training_plan(X, y, batch_size, lr, model_params):
            # inject params into model
            set_model_params(model, model_params)

            # forward pass
            logits = model.forward(X)
    
            # loss
            loss = loss_func(logits, y, batch_size)
    
            # backprop
            loss.backward()

            # step
            updated_params = [
                optim_func(param, lr=lr)
                for param in model_params
            ]
    
            # accuracy
            pred = th.argmax(logits, dim=1)
            target = th.argmax(y, dim=1)
            acc = pred.eq(target).sum().float() / batch_size

            return (
                loss,
                acc,
                *updated_params
            )
    
    # Create dummy input parameters to make the trace, build model
    model_params = [param.data for param in model.parameters()]  # raw tensors instead of nn.Parameter
    lr = th.tensor([0.01])
    batch_size = th.tensor([3.0])
    
    training_plan.build(X, y, batch_size, lr, model_params, trace_autograd=True)
    
    return model_params, training_plan


# Define standard averaging plan
def def_avg_plan(model_params, func=None):
    if func is not None:
        @sy.func2plan()
        def avg_plan(avg, item, num):
            func(avg, item, num)
    else:
        @sy.func2plan()
        def avg_plan(avg, item, num):
            new_avg = []
            for i, param in enumerate(avg):
                new_avg.append((avg[i] * num + item[i]) / (num + 1))
            return new_avg

    # Build the Plan
    avg_plan.build(model_params, model_params, th.tensor([1.0]))
    
    return avg_plan


def artificien_connect(dataset_id, model_id, password):
    """ Function to connect to artificien PyGrid node """
    #api authentication using cognito
    count = 0
    try:
        u = Cognito(userPoolId, clientId, username=os.environ['JUPYTERHUB_USER'])
        u.authenticate(password=password)
    except:
        exit({"Error": "Failed to authenticate User"})

    accessId = u.access_token
    # PyGrid masterNode address from constants.py
    masterNodeAddy = 'http://' + masterNode + '/create'

    # create header with cognito access token and json content specifier
    headers = CaseInsensitiveDict()
    headers["Authorization"] = "Bearer " + accessId
    headers["Content-Type"] = "application/json"
    node = {"dataset_id": dataset_id, "model_id": model_id}
    resp = requests.post(masterNodeAddy, headers=headers, json=node)
    resp = resp.json()
    print(resp)
    #print(resp.get('status'))
    # ping masternode until the ready status is recieved (the node is deployed)
    while resp.get('status') != 'ready':
        count = count + 1
        if 'error' in resp:
            exit({'error': 'failed to connect'})
        time.sleep(30)
        resp = requests.post(masterNodeAddy, headers=headers, json=node)
        resp = resp.json()
        print(resp.get('status'))

    # grab node url from response
    nodeURL = resp.get('nodeURL')
    nodeURL = str(nodeURL) + ':5000'

    #if node has just been deployed above, wait a few minutes until it's fully deployed
    if(count != 0):
        print("Your model's trainers are waking from a long slumber. This may be a few minutes")
        time.sleep(180)

    # connect to grid
    grid = ModelCentricFLClient(id=dataset_id, address=nodeURL, secure=False)
    grid.connect()  # These name/version you use in worker
    # return grid
    return grid

def send_model(name, version, batch_size, learning_rate, max_updates, model_params, training_plan, avg_plan, dataset_id, password):
    """ Function to send model to node """

    # Add username to the model name so as to avoid conflicts across users
    name = name + '-' + version + '-' + os.environ['JUPYTERHUB_USER']
    table = dynamodb.Table('model_table')

    table.put_item(
        Item={
            'model_id': name,
            'active_status': 1,
            'version': version,
            'dataset': dataset_id,
            'date_submitted': str(date.today()),
            'owner_name': str(os.environ['JUPYTERHUB_USER']),
            'percent_complete': 0,
        }
    )

    grid = artificien_connect(dataset_id, name, password)

    client_config = {
        "name": name,
        "version": version,
        "batch_size": batch_size,
        "lr": learning_rate,
        "max_updates": max_updates  # custom syft.js option that limits number of training loops per worker
    }

    server_config = {
        "min_workers": 5,
        "max_workers": 5,
        "pool_selection": "random",
        "do_not_reuse_workers_until_cycle": 6,
        "cycle_length": 28800,  # max cycle length in seconds
        "num_cycles": 5,  # max number of cycles
        "max_diffs": 1,  # number of diffs to collect before avg
        "minimum_upload_speed": 0,
        "minimum_download_speed": 0,
        "iterative_plan": True  # tells PyGrid that avg plan is executed per diff
    }

    model_params_state = State(
        state_placeholders=[
            PlaceHolder().instantiate(param)
            for param in model_params
        ]
    )

    response = grid.host_federated_training(
        model=model_params_state,
        client_plans={'training_plan': training_plan},
        client_protocols={},
        server_averaging_plan=avg_plan,
        client_config=client_config,
        server_config=server_config
    )

    return print("Host response:", response)

def sendWsMessage(data):
    """ Helper function to make WS requests """

    ws = create_connection('ws://' + gridAddress)

    ws.send(json.dumps(data))
    message = ws.recv()
    return json.loads(message)


def check_hosted_model(name, version):
    
    cycle_request = {
        "type": "model-centric/cycle-request",
        "data": {
            "worker_id": auth_response['data']['worker_id'],
            "model": name,
        
            "version": version,
            "ping": 1,
            "download": 10000,
            "upload": 10000,
        }
    }
    cycle_response = sendWsMessage(cycle_request)
    print('Cycle response:', json.dumps(cycle_response, indent=2))

    worker_id = auth_response['data']['worker_id']
    request_key = cycle_response['data']['request_key']
    model_id = cycle_response['data']['model_id'] 
    training_plan_id = cycle_response['data']['plans']['training_plan']
    
    # Model
    req = requests.get(f"http://{gridAddress}/model-centric/get-model?worker_id={worker_id}&request_key={request_key}&model_id={model_id}")
    model_data = req.content
    pb = StatePB()
    pb.ParseFromString(req.content)
    model_params_downloaded = protobuf.serde._unbufferize(hook.local_worker, pb)
    print("Params shapes:", [p.shape for p in model_params_downloaded.tensors()])
    
    # Plan "list of ops"
    req = requests.get(f"http://{gridAddress}/model-centric/get-plan?worker_id={worker_id}&request_key={request_key}&plan_id={training_plan_id}&receive_operations_as=list")
    pb = PlanPB()
    pb.ParseFromString(req.content)
    plan_ops = protobuf.serde._unbufferize(hook.local_worker, pb)
    print(plan_ops.code)
    print(plan_ops.torchscript)
    
    # Plan "torchscript"
    req = requests.get(f"http://{gridAddress}/model-centric/get-plan?worker_id={worker_id}&request_key={request_key}&plan_id={training_plan_id}&receive_operations_as=torchscript")
    pb = PlanPB()
    pb.ParseFromString(req.content)
    plan_ts = protobuf.serde._unbufferize(hook.local_worker, pb)
    print(plan_ts.code)
    print(plan_ts.torchscript.code)
    
    # Plan "tfjs"
    req = requests.get(f"http://{gridAddress}/model-centric/get-plan?worker_id={worker_id}&request_key={request_key}&plan_id={training_plan_id}&receive_operations_as=tfjs")
    pb = PlanPB()
    pb.ParseFromString(req.content)
    plan_tfjs = protobuf.serde._unbufferize(hook.local_worker, pb)
    print(plan_tfjs.code)