from __future__ import unicode_literals, print_function, division
from io import open
import glob
import torch
import torch.nn as nn
import time
import math
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
"""

m = nn.Linear(5,10,bias=False)
print(m)
input = torch.ones(3,5)
output = m(input)
print(input)
print(output)


"""
with open('testset.pkl', 'rb') as f:
    data = pickle.load(f)
print(data[all_categories[1]])