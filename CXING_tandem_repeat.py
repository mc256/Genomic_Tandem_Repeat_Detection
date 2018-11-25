#!/usr/bin/env python
# coding: utf-8

# # Tandem Repeat Deep Learning Based Detector v2
# 

# We need to import some library first.

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
#get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
#from IPython.display import clear_output
import numpy as np
import math


# In[3]:


import hashlib
import datetime


# Configuration

# In[4]:


EPOCH = 10
LR = 0.001          
SAMPLE_SIZE = 200000
BATCH_SIZE = 200
TANDEM_LENGTH = 31
TEST_SIZE = 150
SEQUENCE_LENGTH = 64
MD5_LENGTH = 16


# ## Tandem Repeat Sequence Generator
# 
# This generator can generate sequence with or without tandem repeats.

# In[5]:


class TandemRepeatDataset(torch.utils.data.Dataset):
    
    init_mat = [
         [1,0,0,0],
         [0,1,0,0],
         [0,0,1,0],
         [0,0,0,1],
    ]
    
    def __init__(self, 
                 salt = b'THIS SHOULD BE SALT', 
                 sample_size = SAMPLE_SIZE,
                 noise_ratio = 0
                ):
        super(TandemRepeatDataset,self).__init__()
        self.salt_a, self.salt_b = salt[:len(salt)//2], salt[len(salt)//2:]
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
                x.append([1,0,0,0])
            elif ch == 'T':
                x.append([0,1,0,0])
            elif ch == 'C':
                x.append([0,0,1,0])
            else:
                x.append([0,0,0,1])
        return torch.tensor(x).float(), torch.tensor(size)
                
        
    def __getitem__(self, idx):
        sequence = []
        random_hash = hashlib.new("md5",self.salt_a + str(idx).encode('utf-8')).digest()
        control_hash = hashlib.new("md5",self.salt_b + str(idx).encode('utf-8')).digest()
        #noise_hash = hashlib.new("md5",self.salt_b + str(idx).encode('utf-8') + self.salt_a).digest()
        #################################################################################
        for b in random_hash:
            sequence.append(self.convert(b))
            sequence.append(self.convert(b >> 2))
            sequence.append(self.convert(b >> 4))
            sequence.append(self.convert(b >> 6))
        #################################################################################    
        repeat_len = control_hash[0] % (TANDEM_LENGTH + 1) #0-32
        repeat_offset = control_hash[1] % TANDEM_LENGTH #0-31
        #noise_len = math.floor(repeat_len * self.noise_ratio)            
        #noise_offset = control_hash[2] % SEQUENCE_LENGTH
        #print(repeat_len, repeat_offset, noise_len, noise_offset)
        #################################################################################   
        if repeat_len != 0:
            common_offset = repeat_len - (repeat_offset % repeat_len)
            #print(common_offset)
            for i in range(0,SEQUENCE_LENGTH):
                sequence[i] = sequence[(i + common_offset)% repeat_len + repeat_offset]
            #for i in range(0,noise_len):
            #    sequence[(i + noise_offset)% SEQUENCE_LENGTH] = self.convert(noise_hash[i // 4] >> (i % 4 * 2))
        #################################################################################
        return torch.FloatTensor(sequence), torch.tensor(repeat_len)
    
                            


# In[6]:


#trDS = TandemRepeatDataset(b'asdfasasdf1wqerqwer')
#TandemRepeatDataset.data_to_string(trDS[54654])


# In[7]:


#                    CAT
#TTAGCAGAAAGACAAATATGCAT
#TTAGCAGAAAGACAAATATGCAT
#TTAGCAGAAAGACAA


# ### Utility Methods

# In[8]:


def display_result(confusion):
    # Plot
    #clear_output(wait=True)
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


# In[9]:


def save():
    torch.save(rnn, 
               "./tandem_repeat"+datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')+
               "--SIZE"+str(TANDEM_LENGTH)+
               "-SLEN"+str(SEQUENCE_LENGTH)+
               "v3.pkl")


# ## Neural Network
# 
# I choose to use LTSM here, the input size is the types of the nucleotides. The time domain is the length of the sequence. The output size is the possible tandem repeat $size + 1$ where the $0$ means no tandem repeat.

# In[10]:


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(     
            input_size=4,       # 4 nucleotides
            hidden_size=128,     # rnn hidden unit
            num_layers=1,       # RNN layers
            batch_first=True,   # (batch, time_step, input_size)
        )

        self.out = nn.Linear(128, TANDEM_LENGTH + 1)    # output layer

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None: hidden state uses 0

        # Pick the last r_out
        out = self.out(r_out[:, -1, :]) # choose the last output # (batch, time step, input)
        return out

rnn = RNN()#.cuda()
rnn = rnn.cuda()
print(rnn)

# <span style="color:red">You could use the follow block of code to load pre-learned model.</span> (No worries if it shows UserWarning)

# In[11]:


#rnn = torch.load("./tandem_repeat2018-11-13-15-10-06--SIZE31-SLEN64v3.pkl") # size 16


# ___
# ___
# 

# # START LEARNING

# I use Adam optimizer for this learning.

# In[27]:


optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted


# In[30]:


trDS = TandemRepeatDataset(b'K34JL1dfasd22dfeG5jVasdl21010nzv')
trDS_Testing = TandemRepeatDataset(b'hav23123412d53DF23orld',TEST_SIZE)


# In[31]:


train_loader = Data.DataLoader(dataset=trDS, batch_size=BATCH_SIZE, shuffle=True)


# In[32]:


loss_history_rnn = []


# In[33]:


for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):   # gives batch data
        x = x.cuda()
        y = y.cuda()
        b_x = x.view(-1, SEQUENCE_LENGTH, 4)   # reshape x to (batch, time_step, input_size)
        b_y = y
        
        
        
        output = rnn(b_x)               # rnn output
        loss = loss_func(output, y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        
        loss = loss.cpu()
        loss_history_rnn.append(loss.data.numpy())
        
        if step % 100 == 0:
            #############################
            # Test
            confusion = torch.zeros(TANDEM_LENGTH + 1, TANDEM_LENGTH + 1)
            for idx in range (0,TEST_SIZE):
                x, y = trDS_Testing[idx]
                x=x.cuda()
                y=y.cuda()
                guess_output = rnn (x.view(-1, SEQUENCE_LENGTH, 4))
                predict_n, predict_i = guess_output.topk(1)
                confusion[y][predict_i] += 1    

            for i in range(TANDEM_LENGTH +1):
                confusion[i] = confusion[i] / confusion[i].sum()
            #############################
            #############################
            print('Epoch: ', epoch, " Step:", step)
        if step == 0:        
            display_result(confusion)


# In[19]:

display_result(confusion)
save()





