import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import math
import numpy as np
import hashlib
import datetime
from Bio import SeqIO


class Node:
    leaves = []
    nodes = []
    value = ""

    def __init__(self):
        self.leaves = []
        self.nodes = []
        self.value = ""
        pass

    def checkNodes(self, st):
        for idx in range(0, len(self.nodes)):
            if self.nodes[idx].value == st[0]:
                self.nodes[idx].addSuffix(st[1:])
                return True
        return False

    def checkLeaves(self, st):
        for idx in range(0, len(self.leaves)):
            leaf = self.leaves[idx]
            if leaf[0] == st[0]:
                node = Node()
                node.value = leaf[0]
                node.addSuffix(st[1:])
                node.addSuffix(leaf[1:])
                self.nodes.append(node)
                del self.leaves[idx]
                return
        self.leaves.append(st)

    def addSuffix(self, st):
        if len(st) == 0 or st == "":
            return
        else:
            if not self.checkNodes(st):
                self.checkLeaves(st)

    def getLongestRepeatedSubString(self):
        str = ""
        for idx in range(0, len(self.nodes)):
            temp = self.nodes[idx].getLongestRepeatedSubString()
            if len(temp) > len(str):
                str = temp
        return self.value + str


class LongestRepeatSubSequence:
    def __init__(self, sequence=""):
        self.sequence = sequence
        self.root = Node()
        for idx in range(0, len(sequence)):
            self.root.addSuffix(sequence[idx:])

    def get_LRS(self):
        return self.root.getLongestRepeatedSubString()


EPOCH = 10
LR = 0.001
SAMPLE_SIZE = 200000
BATCH_SIZE = 200
TANDEM_LENGTH = 7
TEST_SIZE = 150
SEQUENCE_LENGTH = 32
MD5_LENGTH = 16

class RepeatRNN(nn.Module):
    def __init__(self):
        super(RepeatRNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=4,       # 4 nucleotides
            hidden_size=128,     # rnn hidden unit
            num_layers=2,       # RNN layers
            batch_first=True,   # (batch, time_step, input_size)
        )

        self.out = nn.Linear(128, TANDEM_LENGTH + 1)    # output layer

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)   # None: hidden state uses 0

        out = self.out(r_out[:, -1, :]) # choose the last output # (batch, time step, input)
        return out

rnn = RepeatRNN()
rnn = torch.load("./temp/tandem_repeat2018-12-06-17-41-42wo_n_s_LARG_NBt8.pkl") # no noise
rnn = rnn.cuda()
print(rnn)


def string_to_data(seq):
    x = []
    for ch in seq:
        if ch == 'A':
            x.append([1, 0, 0, 0])
        elif ch == 'T':
            x.append([0, 1, 0, 0])
        elif ch == 'C':
            x.append([0, 0, 1, 0])
        elif ch == 'G':
            x.append([0, 0, 0, 1])
        else:
            x.append([0, 0, 0, 0])
    return torch.tensor(x).float()


#tracking_heat_map = []
#top_val = []
seq_records  = SeqIO.parse("D:/owncloud/Education/EECS4425+/Assignment/a1/DSM4304.fasta","fasta")
#seq_records  = SeqIO.parse("C:/Users/masterchan/Desktop/CM010885.1.fasta","fasta")
#seq_records  = SeqIO.parse("D:/AF047825.1.fasta","fasta")
for record in seq_records:
    seq = record.seq
    count = 0
    #print(repr(seq))
    for idx in range(0,len(seq) - 31, SEQUENCE_LENGTH):
        #print(seq[idx:idx+SEQUENCE_LENGTH])
        if seq[idx] == "N":
            continue
        vector = string_to_data(seq[idx:idx+SEQUENCE_LENGTH])
        k = rnn(vector.cuda().view(-1, SEQUENCE_LENGTH, 4))
        y_axis = k.cpu().detach().squeeze().tolist()
        #tracking_heat_map.append(y_axis)
        k = k.cpu()
        signal, predicted_result = k.topk(5)
        predicted_result = predicted_result.numpy()
        #print(predicted_result)
        if predicted_result[0][0] != 0:
            #signal = signal.numpy()
            print("%10d -->[%2d %2d %2d %2d]" % (idx, predicted_result[0][0], predicted_result[0][1], predicted_result[0][2], predicted_result[0][3]), end='  ')
            print(seq[idx:idx + SEQUENCE_LENGTH],end=" ")
            test = LongestRepeatSubSequence(seq[idx:idx + SEQUENCE_LENGTH])
            print("Repeat:",test.get_LRS(), end="")
            print()
            count += 1

print("total %d " % count)