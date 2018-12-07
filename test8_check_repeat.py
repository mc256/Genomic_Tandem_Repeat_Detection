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
SAMPLE_SIZE = 6000000
BATCH_SIZE = 2000
TANDEM_LENGTH = 15
TEST_SIZE = 150
SEQUENCE_LENGTH = 32
MD5_LENGTH = 16

class TandemRepeatDataset(torch.utils.data.Dataset):
    init_mat = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]

    lookup_table = [
        0,
        4, #1
        12,
        60,
        240,
        1020,
        4020,
        16380,
        65280,
        262080,
        1047540,
        4194300,
        16772880,
        67108860,
        268419060,
        1073740740
    ]

    pow_quick_check = [
        1,
        256,
        65536,
        16777216,
        4294967296,
    ]

    def __init__(self,
                 salt=b'THIS SHOULD BE SALT',
                 sample_size=SAMPLE_SIZE,
                 noise_ratio=0.0,
                 trim_to=SEQUENCE_LENGTH,
                 biased = False
                 ):
        super(TandemRepeatDataset, self).__init__()
        self.salt_a, self.salt_b = salt[:len(salt) // 2], salt[len(salt) // 2:]
        self.sample_size = sample_size
        self.noise_ratio = noise_ratio
        self.trim_to = trim_to
        self.biased = biased

    def __len__(self):
        return self.sample_size

    def convert(self, ch):
        t = ch % 4
        return self.init_mat[t]

    def data_to_string(data):
        x, y, z = data
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
        print(dna_sequence, " -- repeat size: ", y.numpy(), " -- noise size: ", z.numpy())

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
        #noise_hash = hashlib.new("md5", self.salt_b + str(idx).encode('utf-8') + self.salt_a).digest()
        #################################################################################
        for b in random_hash:
            sequence.append(self.convert(b))
            sequence.append(self.convert(b >> 2))
            sequence.append(self.convert(b >> 4))
            sequence.append(self.convert(b >> 6))
        #################################################################################
        repeat_len = 0
        if self.biased:
            repeat_len = control_hash[0] % (TANDEM_LENGTH + 1)
        else:
            accumulate = 0
            for item in range(0,4):
                accumulate = accumulate + control_hash[item] * self.pow_quick_check[item]
            for item in range(1,16):
                if accumulate < self.lookup_table[item]:
                    repeat_len = item
                    break
        repeat_offset = control_hash[15] % TANDEM_LENGTH  # 0-31
        #print(repeat_len, repeat_offset)
        #################################################################################
        # Compute the noise ratio
        ################ May violate "equally likely rule"
        #noise_max_len = math.floor(SEQUENCE_LENGTH * self.noise_ratio)
        #if (noise_max_len != 0):
        #    noise_len = (control_hash[3] + control_hash[4] + control_hash[5]) % noise_max_len  # how long the noise pattern is
        #    noise_offset = control_hash[2] % SEQUENCE_LENGTH  # where the noise pattern starts
        #else:
        #    noise_len = 0
        #    noise_offset = 0
        # print(noise_len, noise_offset)
        #################################################################################
        if repeat_len != 0:
            common_offset = repeat_len - (repeat_offset % repeat_len)
            for i in range(0, SEQUENCE_LENGTH):
                sequence[i] = sequence[(i + common_offset) % repeat_len + repeat_offset]
            #for i in range(0, noise_len):
            #    sequence[(i + noise_offset) % SEQUENCE_LENGTH] = self.convert(noise_hash[i // 4] >> (i % 4 * 2))
        #################################################################################
        return torch.tensor(sequence[:self.trim_to]).float(), torch.tensor(repeat_len) #, torch.tensor(noise_len)

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


def save(comment=""):
    torch.save(rnn,
               "./temp/tandem_repeat"+datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')+comment+
               "t8.pkl")


class RepeatRNN(nn.Module):
    def __init__(self):
        super(RepeatRNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=4,       # 4 nucleotides
            hidden_size=128,     # rnn hidden unit
            num_layers=3,       # RNN layers
            batch_first=True,   # (batch, time_step, input_size)
        )

        self.out = nn.Linear(128, TANDEM_LENGTH + 1)    # output layer

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)   # None: hidden state uses 0

        out = self.out(r_out[:, -1, :]) # choose the last output # (batch, time step, input)
        return out

rnn = RepeatRNN()
rnn = rnn.cuda()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted


ANOTHER_TESTER = 1000

confusion = torch.zeros(TANDEM_LENGTH + 1, TANDEM_LENGTH + 1)


data_source = TandemRepeatDataset(b'salt with no noise. HAVE FUN asdf', noise_ratio=0, biased=True)
train_loader = Data.DataLoader(dataset=data_source, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(1,6):
    print("EPOCH: %2d " % epoch, end="")
    for step, (x, y) in enumerate(train_loader):  # gives batch data
        x = x.cuda()
        y = y.cuda()
        b_x = x.view(-1, SEQUENCE_LENGTH, 4)  # reshape x to (batch, time_step, input_size)
        b_y = y
        output = rnn(b_x)  # rnn output
        loss = loss_func(output, y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        loss_history_rnn.append(loss.cpu().data.numpy())
        if step % 100 == 0:
            print(".", end="")
    #############################
    # Test
    confusion = torch.zeros(TANDEM_LENGTH + 1, TANDEM_LENGTH + 1)
    testing_source = TandemRepeatDataset(b'hav2231323{}d' + str(step).encode('utf-8') + str(epoch).encode('utf-8'),
                                         ANOTHER_TESTER, biased=True)
    for idx in range(0, TEST_SIZE):
        x, y = testing_source[idx]
        x = x.cuda()
        y = y.cuda()
        guess_output = rnn(x.view(-1, SEQUENCE_LENGTH, 4))
        predict_n, predict_i = guess_output.topk(1)
        confusion[y][predict_i] += 1
    for i in range(TANDEM_LENGTH + 1):
        confusion[i] = confusion[i] / confusion[i].sum()
    #############################
    display_result(confusion)
    print(".")
    save("epoch-%d_NB"%epoch)

save("wo_n_s_LARG_NB")