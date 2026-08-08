import os
import requests
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass

class FeedForwardCatalog(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config        

    def mlp_with_relu(self):
        model = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_model * 4),
                nn.ReLU(),
                nn.Linear(self.config.d_model * 4, self.config.d_model)
            )

        return model

    def mlp_with_gelu(self):
        model = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_model * 4),
                nn.GELU(),
                nn.Linear(self.config.d_model * 4, self.config.d_model)
            )

        return model

    def mlp_with_swiglu(self):
        model = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_model * 4),
                nn.GELU(), # have to add swiglu functionality
                nn.Linear(self.config.d_model * 4, self.config.d_model)
            )

        return model
