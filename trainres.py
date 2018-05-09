from __future__ import unicode_literals, print_function, division
from io import open
import glob
import numpy as np
import torch
import torch.nn as nn
import time
import math
import random
import pickle
def findFiles(path): return glob.glob(path)
import unicodedata
import string


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines




n_categories = len(all_categories)
testset = {}
trainset= {}
testsize = 10


for i in range(n_categories):

    random.shuffle(category_lines[all_categories[i]])

    testset[all_categories[i]] = category_lines[all_categories[i]][0:testsize]

    trainset[all_categories[i]] = category_lines[all_categories[i]][testsize:-1]



##########################################################
f = open("testset.pkl","wb")
pickle.dump(testset,f)
f.close()

f = open("trainset.pkl","wb")
pickle.dump(trainset,f)
f.close()
f = open("allcategories.pkl","wb")
pickle.dump(all_categories,f)
f.close()


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

###########################################################

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

n_hidden = 500
rnn = RNN(n_letters, n_hidden, n_categories)



def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(trainset[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


criterion = nn.NLLLoss()

###########################################################

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    hiddden = hidden.cuda()
    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()
    z = 0

    for p in rnn.parameters():
        if z!=0 and z!=1:
            p.data.add_(-learning_rate, p.grad.data)

        z=z+1
        #if z !=0 and z !=1:
      #z=z+1

    return output, loss.item()


n_iters = 200000
print_every = 5000
plot_every = 1000

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()
correct = 0


for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess

    guess, guess_i = categoryFromOutput(output)
    if guess == category and iter>99000:
        correct = correct+1

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
        print(iter, iter / n_iters * 100, timeSince(start),loss)

#model = rnn()

#model.save_state_dict('resv.pt')
torch.save(rnn,'resv.pt')
print('acc')
print(correct/1000)




