import os
import requests
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass

class InitParams():
    def __init__(self, config):
        self.initialilized = None
