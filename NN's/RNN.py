import torch
import torch.nn as nn

device = torch.device("cuda")


# Model
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.Rnn = nn.RNN(1, 64)  # 1 input feature and 64 output features
        self.Linear = nn.Linear(64, 1)

    def forward(self, x, h):
        x, h = self.Rnn(x, h)  # x is the input sequence and h representative for hidden state.
        x = self.Linear(x)

        return x, h


model = RNN().to(device)

# Data
Input = torch.rand(10, 1, 1).to(device)
Target = torch.rand(10, 1, 1).to(device)
hidden_state = torch.zeros(1, 1, 64).to(device)  # 1 layer, 1 batch size and 64 output features.

learning_rate = 0.01
num_epochs = 1000

Criterion = nn.L1Loss()  # L1Loss seems to the best loss function for this model
Optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train
for epoch in range(num_epochs):
    Optimizer.zero_grad()
    Outputs, hidden_state = model(Input, hidden_state.detach())  # produce an output sequence and new hidden state
    loss = Criterion(Outputs, Target)
    loss.backward()
    Optimizer.step()

    if epoch % 5 == 0:
        print(f'Epoch: [{epoch + 1} / {num_epochs}], Loss: {loss:.5f}')
