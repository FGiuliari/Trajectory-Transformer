# -*- coding: utf-8 -*-
# date: 2018-11-29 20:03
import torch.nn as nn
from torch.nn.functional import log_softmax


class Generator(nn.Module):
    """
    Define standard linear + softmax generation step.
    """

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x)
