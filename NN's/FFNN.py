# Import the modules
import torch
import torch.nn as nn

# Assign the device
device = torch.device("cuda")


# Define the model
class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()
        self.Fc1 = nn.Linear(2, 64)
        self.Fc2 = nn.Linear(64, 128)
        self.Fc3 = nn.Linear(128, 2)
        self.ReLu = nn.ReLU()

    def forward(self, x):
        x = self.ReLu(self.Fc1(x))
        x = self.ReLu(self.Fc2(x))
        x = self.Fc3(x)
        return x


model = FFNN().to(device)  # Move the model to the GPU

Input = torch.rand(2).to(device)  # Random tensor inputs

Target = torch.rand(2).to(device)  # Random target tensor outputs

num_epochs = 100
Learning_rate = 0.01
Criterion = nn.MSELoss()  # Loss function
Optimizer = torch.optim.SGD(model.parameters(), lr=Learning_rate)  # model.parameters() provides learnable parameters

# Training Loop
for epoch in range(num_epochs):
    Optimizer.zero_grad()  # Clears the optimizer's gradient buffers in each epoch
    Outputs = model(Input)
    loss = Criterion(Outputs, Target)
    loss.backward()
    Optimizer.step()  # Updates the model parameters

    if epoch % 5 == 0:  # Prints the current epoch number and loss value
        print(f'Epoch: [{epoch + 1} / {num_epochs}], Loss: {loss:.5f}')
