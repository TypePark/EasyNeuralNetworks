import torch
import torch.nn as nn


device = torch.device("cuda")

# Model
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(1, 64)  # 1 input feature and 64 hidden features
        self.lstm2 = nn.LSTM(64, 128)
        self.lstm3 = nn.LSTM(128, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x, h_list, c_list): # I used a list for scalibility purposes
        x, (h1, c1) = self.lstm1(x, (h_list[0], c_list[0]))
        x, (h2, c2) = self.lstm2(x, (h_list[1], c_list[1]))
        x, (h3, c3) = self.lstm3(x, (h_list[2], c_list[2]))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x, [h1, h2, h3], [c1, c2, c3]


model = LSTM().to(device)

# Data
input_data = torch.rand(10, 1, 1).to(device)
target = torch.rand(10, 1, 1).to(device)

hidden_states = [torch.zeros(1, input_data.size(1), 64).to(device),
                 torch.zeros(1, input_data.size(1), 128).to(device),
                 torch.zeros(1, input_data.size(1), 256).to(device)]  # List of hidden states for each LSTM layer

cell_states = [torch.zeros(1, input_data.size(1), 64).to(device),
               torch.zeros(1, input_data.size(1), 128).to(device),
               torch.zeros(1, input_data.size(1), 256).to(device)]  # List of cell states for each LSTM layer

learning_rate = 0.001
num_epochs = 1000

criterion = nn.SmoothL1Loss()  # SmoothL1Loss is performing better than L1Loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # and Adam is performing better than SGD
#Sometimes can 0 the loss under 1000 epoch but most of the time stucks at a local minumum


# Train
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs, hidden_states, cell_states = model(input_data, [h.detach() for h in hidden_states],[c.detach() for c in cell_states])

    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f'Epoch: [{epoch + 1} / {num_epochs}], Loss: {loss:.5f}')
