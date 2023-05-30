#%% md

#%%

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device("cuda:0") # GPU에서 실행하려면 이 주석을 제거하세요

class nnApproximation(nn.Module):
    def __init__(self, input_dim):
        super(nnApproximation, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim+1)
        )
    def forward(self, x):
        return self.layers(x)
print("class created")


learning_rate = 0.00001
num_epochs = 1000
num_samples = 100
test_samples = 1000


# Define models
m4=nnApproximation(4)
m8=nnApproximation(8)
m16=nnApproximation(16)
m32=nnApproximation(32)

# Define the loss function and optimizer
loss_function = nn.L1Loss()

op4 = torch.optim.Adam(m4.parameters(), lr=learning_rate)
op8 = torch.optim.Adam(m8.parameters(), lr=learning_rate)
op16 = torch.optim.Adam(m16.parameters(), lr=learning_rate)
op32 = torch.optim.Adam(m32.parameters(), lr=learning_rate)

print("model created")

#%% md

#%%

# Training loop for param4
print("starting training model with 4 params")
loss4list=list()
for epoch in range(num_epochs):
    # Generate random input
    X = torch.rand(num_samples, 4).to(device)*10
    current_loss = 0.0
    for t in range(num_samples):
        Xs=X[t]
        op4.zero_grad()
        pred=m4(Xs)
        if torch.isnan(pred).any():
            print("nan detected")
        log_sum = torch.log(torch.sum(Xs))
        log_individual = torch.sum(torch.log(Xs) * pred[:-1])
        log_constant = pred[-1]
        loss = loss_function(torch.pow(torch.abs(log_sum - log_individual - log_constant), 2), torch.zeros(4))
        loss.backward()
        op4.step()
        current_loss += loss.item()
        if t % 10 == 0:
            #print('Loss after mini-batch %5d: %.3f' %
            #      (t + 1, current_loss / 500))
            loss4list.append(current_loss)
            current_loss = 0.0
print(len(loss4list))


X_test = torch.rand(test_samples,4).to(device)*10
with torch.no_grad():
    Y_pred = m4(X_test)
    LHS=list()
    RHS=list()
    for t in range(test_samples):
        LHS.append(torch.log(torch.sum(X_test[t])))
        RHS.append(torch.sum(torch.log(X_test[t])*Y_pred[t][:-1]) + Y_pred[t][-1])
    LHS=torch.tensor(LHS)
    RHS=torch.tensor(RHS)
    print(LHS)
    print(RHS)
    #plt.plot(X_test.numpy(), LHS.numpy(),color='r',label='LHS')
    #plt.plot(X_test.numpy(), RHS.numpy(),color='r',label='LHS')
    plt.scatter(LHS.numpy(),RHS.numpy())
    plt.legend()
    plt.show()

#%% md


#%%

with torch.no_grad():
    m4.eval()
    torch.jit.script(m4).save('m4.pt')
    m8.eval()
    torch.jit.script(m8).save('m8.pt')
    m16.eval()
    torch.jit.script(m16).save('m16.pt')
    m32.eval()
    torch.jit.script(m32).save('m32.pt')

#%%

print(X_test.reshape(4,-1).size(dim=0))
print(X_test.size(dim=0))


#%%

test=torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12])
print(test.reshape(4,-1))
print(test.reshape(-1,4).size(dim=0))
