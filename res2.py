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
import unicodedata
import string
import pickle
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

rnn = torch.load('resv.pt')

with open('testset.pkl', 'rb') as f:
    testset = pickle.load(f)
with open('allcategories.pkl', 'rb') as f:
    all_categories = pickle.load(f)

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)



def letterToIndex(letter):
    return all_letters.find(letter)

def lineToTensor(line):

    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1

    return tensor


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i



def generate(k,l):

    category = all_categories[k]
    print(category)
    line = testset[category][l]
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    print(line)
    return category, line, category_tensor, line_tensor


def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

correct = 0

for i in range(180):
    k = int(i/10)
    l = i%10
    print(k)

    category, line, category_tensor, line_tensor = generate(k,l)
    output = evaluate(line_tensor)


    guess, guess_i = categoryFromOutput(output)
    if guess == category :
        correct = correct+1


print(correct/180)

