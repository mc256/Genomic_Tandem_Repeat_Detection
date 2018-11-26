import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import math
import numpy as np
import hashlib
import datetime
from Bio import SeqIO

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
rnn = torch.load("./temp/tandem_repeat2018-11-26-11-19-18w0.5n_st6.pkl") # with 50% noise
#rnn = torch.load("./temp/tandem_repeat2018-11-25-23-00-41w_n_st6.pkl") # with noise
#rnn = torch.load("./temp/tandem_repeat2018-11-25-22-55-46wo_n_st6.pkl") # no noise
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
        else:
            x.append([0, 0, 0, 1])
    return torch.tensor(x).float()


#tracking_heat_map = []
#top_val = []
seq_records  = SeqIO.parse("D:/owncloud/Education/EECS4425+/Assignment/a1/DSM4304.fasta","fasta")
for record in seq_records:
    seq = record.seq
    #print(repr(seq))
    for idx in range(0,len(seq), SEQUENCE_LENGTH):
        #print(seq[idx:idx+SEQUENCE_LENGTH])
        vector = string_to_data(seq[idx:idx+SEQUENCE_LENGTH])
        k = rnn(vector.cuda().view(-1, SEQUENCE_LENGTH, 4))
        y_axis = k.cpu().detach().squeeze().tolist()
        #tracking_heat_map.append(y_axis)
        y_n, y_i = k.topk(1)
        #top_val.append(y_i.cpu().item())
        predicted_result = y_i.cpu().item()
        if predicted_result != 0:
            print(idx, "-->", predicted_result)
            print(seq[idx:idx + SEQUENCE_LENGTH])
"""
fig = plt.figure(figsize=(15,2))
ax=fig.add_subplot(1,1,1)
cax = ax.matshow(np.transpose(np.array(tracking_heat_map, int)))
x = 0
for item in top_val:
    ax.text(x, item, 'x', va='center', ha='center', color='red')
    x+=1
fig.colorbar(cax)
plt.xlabel('Sequence location / 64')
plt.ylabel('Predict Repeat Size')
ax.xaxis.set_label_position('top')
plt.show()
"""