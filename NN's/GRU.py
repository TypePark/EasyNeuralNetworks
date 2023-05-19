# Mostly the same as LSTM, with the only difference being the absence of cell states

import torch
import torch.nn as nn


device = torch.device("cuda")

# Model
class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.gru1 = nn.GRU(1, 64)  # 1 input feature and 64 hidden features
        self.gru2 = nn.GRU(64, 128)
        self.gru3 = nn.GRU(128, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x, h_list): # No cell states in GRU
        x, h1 = self.gru1(x, h_list[0])
        x, h2 = self.gru2(x, h_list[1])
        x, h3 = self.gru3(x, h_list[2])
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x, [h1, h2, h3]



model = GRU().to(device)

# Data
input_data = torch.rand(10, 1, 1).to(device)
target = torch.rand(10, 1, 1).to(device)

hidden_states = [torch.zeros(1, input_data.size(1), 64).to(device),
                 torch.zeros(1, input_data.size(1), 128).to(device),
                 torch.zeros(1, input_data.size(1), 256).to(device)]  # List of hidden states for each GRU layer



learning_rate = 0.001
num_epochs = 1000

criterion = nn.SmoothL1Loss()  # SmoothL1Loss is performing better than L1Loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # and Adam is performing better than SGD
#Sometimes can 0 the loss under 1000 epoch but most of the time stucks at a local minumum ( still true for GRU )


# Train
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs, hidden_states = model(input_data, [h.detach() for h in hidden_states])  # produces an output sequence and new hidden states

    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f'Epoch: [{epoch + 1} / {num_epochs}], Loss: {loss:.5f}')
