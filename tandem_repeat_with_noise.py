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
                 noise_ratio=0.0,
                 trim_to=SEQUENCE_LENGTH
                 ):
        super(TandemRepeatDataset, self).__init__()
        self.salt_a, self.salt_b = salt[:len(salt) // 2], salt[len(salt) // 2:]
        self.sample_size = sample_size
        self.noise_ratio = noise_ratio
        self.trim_to = trim_to

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
        noise_max_len = math.floor(SEQUENCE_LENGTH * self.noise_ratio)
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
        return torch.tensor(sequence[:self.trim_to]).float(), torch.tensor(repeat_len), torch.tensor(noise_len)

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
    torch.save(cnn,
               "./temp/tandem_repeat"+datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')+comment+
               "cnn.pkl")



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
            nn.ReLU(),
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
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=12)
        )
        self.out = nn.Linear(TANDEM_LENGTH + 1,TANDEM_LENGTH)

    def forward(self, x):
        r1 = self.cnn1(x)
        r2 = self.cnn2(r1)
        return self.out(r2)


cnn = NoiseCNN()
cnn = cnn.cuda()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted


ANOTHER_TESTER = 1000
trDS = TandemRepeatDataset(b'K34JL1ferG5jVasdl21010nzv')
trDS_Testing = TandemRepeatDataset(b'hav23123412dg3orld', ANOTHER_TESTER, noise_ratio=0.5)


train_loader = Data.DataLoader(dataset=trDS, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCH):
    for step, (x, y, z) in enumerate(train_loader):  # gives batch data
        x = x.cuda()
        y = y.cuda()
        z = z.cuda()
        b_x = x.view(-1, SEQUENCE_LENGTH, 4)  # reshape x to (batch, time_step, input_size)
        b_y = y

        output = cnn(b_x)  # rnn output
        loss = loss_func(output, y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        loss_history_rnn.append(loss.cpu().data.numpy())

        if step % 100 == 0:
            #############################
            # Test
            confusion = torch.zeros(TANDEM_LENGTH + 1, TANDEM_LENGTH + 1)
            for idx in range(0, TEST_SIZE):
                x, y, z= trDS_Testing[idx]
                x = x.cuda()
                y = y.cuda()
                z = z.cuda()
                guess_output = cnn(x)
                predict_n, predict_i = guess_output.topk(1)
                confusion[y][predict_i] += 1

            for i in range(TANDEM_LENGTH + 1):
                confusion[i] = confusion[i] / confusion[i].sum()
            #############################
            #############################
            print('Epoch: ', epoch, " Step:", step)
        if step == 0:
            display_result(confusion)
            save()
