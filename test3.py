import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import math
import numpy as np
import hashlib
import datetime


EPOCH = 10
LR = 0.001
SAMPLE_SIZE = 200000
BATCH_SIZE = 200
TANDEM_LENGTH = 31
TEST_SIZE = 150
SEQUENCE_LENGTH = 64
MD5_LENGTH = 16
TRIM_SIZE = TANDEM_LENGTH

class TandemRepeatDataset(torch.utils.data.Dataset):
    init_mat = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]

    def __init__(self,
                 salt=b'THIS SHOULD BE SALT',
                 sample_size=SAMPLE_SIZE,
                 noise_ratio=0
                 ):
        super(TandemRepeatDataset, self).__init__()
        self.salt_a, self.salt_b = salt[:len(salt) // 2], salt[len(salt) // 2:]
        self.sample_size = sample_size
        self.noise_ratio = noise_ratio

    def __len__(self):
        return self.sample_size

    def convert(self, ch):
        t = ch % 4
        return self.init_mat[t]

    def data_to_string(data):
        x, y = data
        dna_sequence = ""
        for item in x.numpy():
            if item[0] == 1:
                dna_sequence += "A"
            elif item[1] == 1:
                dna_sequence += "T"
            elif item[2] == 1:
                dna_sequence += "C"
            else:
                dna_sequence += "G"
        print(dna_sequence, " -- repeat size: ", y.numpy())

    def string_to_data(seq, size):
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
        return torch.tensor(x).float(), torch.tensor(size)

    def __getitem__(self, idx):
        sequence = []
        random_hash = hashlib.new("md5", self.salt_a + str(idx).encode('utf-8')).digest()
        control_hash = hashlib.new("md5", self.salt_b + str(idx).encode('utf-8')).digest()
        noise_hash = hashlib.new("md5", self.salt_b + str(idx).encode('utf-8') + self.salt_a).digest()
        #################################################################################
        for b in random_hash:
            sequence.append(self.convert(b))
            sequence.append(self.convert(b >> 2))
            sequence.append(self.convert(b >> 4))
            sequence.append(self.convert(b >> 6))
        #################################################################################
        repeat_len = control_hash[0] % (TANDEM_LENGTH + 1)  # 0-32
        repeat_offset = control_hash[1] % TANDEM_LENGTH  # 0-31
        # print(repeat_len, repeat_offset)
        #################################################################################
        # Compute the noise ratio
        ################ May violate "equally likely rule"
        noise_max_len = math.floor(SEQUENCE_LENGTH * self.noise_ratio);
        if (noise_max_len != 0):
            noise_len = (control_hash[3] + control_hash[4] + control_hash[5]) % noise_max_len  # how long the noise pattern is
            noise_offset = control_hash[2] % SEQUENCE_LENGTH  # where the noise pattern starts
        else:
            noise_len = 0
            noise_offset = 0
        # print(noise_len, noise_offset)
        #################################################################################
        if repeat_len != 0:
            common_offset = repeat_len - (repeat_offset % repeat_len)
            for i in range(0, SEQUENCE_LENGTH):
                sequence[i] = sequence[(i + common_offset) % repeat_len + repeat_offset]
            for i in range(0, noise_len):
                sequence[(i + noise_offset) % SEQUENCE_LENGTH] = self.convert(noise_hash[i // 4] >> (i % 4 * 2))
        #################################################################################
        return torch.tensor(sequence[:TRIM_SIZE]).float(), torch.tensor(repeat_len), torch.tensor(noise_len)


loss_history_rnn = []

def display_result(confusion):
    fig = plt.figure(figsize=(13,5))
    plt.subplot(1,2,1)
    plt.plot(loss_history_rnn)
    plt.title("Training")
    plt.grid(True)
    plt.xlabel('Steps')
    plt.ylabel('Loss')

    ax=fig.add_subplot(1,2,2)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)
    plt.title("Testing")
    plt.xlabel('Predicted Result')
    plt.ylabel('Tandem Repeat Size')

    plt.show()


def save():
    torch.save(rnn,
               "./temp/tandem_repeat"+datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')+
               "-SIZE"+str(TANDEM_LENGTH)+
               "-SLEN"+str(SEQUENCE_LENGTH)+
               "v4.pkl")


class RepeatRNN(nn.Module):
    def __init__(self):
        super(RepeatRNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=4,       # 4 nucleotides
            hidden_size=128,     # rnn hidden unit
            num_layers=1,       # RNN layers
            batch_first=True,   # (batch, time_step, input_size)
        )

        self.out = nn.Linear(128, TANDEM_LENGTH + 1)    # output layer

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)   # None: hidden state uses 0

        out = self.out(r_out[:, -1, :]) # choose the last output # (batch, time step, input)
        return out

rnn = RepeatRNN()
rnn = torch.load("./temp/tandem_repeat2018-11-23-22-19-54-SIZE31-SLEN64v4.pkl")
rnn = rnn.cuda()
print(rnn)

"""
class NoiseCNN(nn.Module):
    def __init__(self):
        self.cnn1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=TANDEM_LENGTH + 1,
                kernel_size=TANDEM_LENGTH // 4,
                stride=1,
                padding=0
            ),
            nn.ReLU,
            nn.MaxPool1d(kernel_size=12)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=TANDEM_LENGTH + 1,
                kernel_size=TANDEM_LENGTH // 4,
                stride=1,
                padding=0
            ),
            nn.ReLU,
            nn.MaxPool1d(kernel_size=12)
        )
        self.out = nn.Linear(TANDEM_LENGTH + 1,TANDEM_LENGTH)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        return self.out(x)

"""

for t_length in range(33, 50):

    ANOTHER_TESTER = 10000
    TRIM_SIZE = t_length
    trDS_Testing_long = TandemRepeatDataset(b'haaasdf234asdg213hb2fersn', ANOTHER_TESTER)

    #############################
    # Test
    confusion = torch.zeros(TANDEM_LENGTH + 1, TANDEM_LENGTH + 1)
    for idx in range (0, ANOTHER_TESTER):
        x, y, z = trDS_Testing_long[idx]
        x = x.cuda()
        y = y.cuda()
        guess_output = rnn(x.view(-1, TRIM_SIZE, 4))
        predict_n, predict_i = guess_output.topk(1)
        confusion[y][predict_i] += 1
        if idx % 1000 == 0:
            print(idx, " out of ", ANOTHER_TESTER)

    for i in range(TANDEM_LENGTH + 1):
        confusion[i] = confusion[i] / confusion[i].sum()

    #############################
    # Plot
    fig = plt.figure(figsize=(11,11))
    ax=fig.add_subplot(1,1,1)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)
    plt.xlabel('Predicted Result')
    plt.ylabel('Actual Tandem Repeat Size')
    plt.title("Sequence Length %d bps" % TRIM_SIZE)
    ax.xaxis.set_label_position('top')
    plt.xticks(np.arange(0,TANDEM_LENGTH + 1).tolist())
    plt.yticks(np.arange(0,TANDEM_LENGTH + 1).tolist())
    #plt.show()
    fig.savefig("./proofs/seq_len%d.png" % TRIM_SIZE)