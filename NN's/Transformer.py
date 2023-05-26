import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda")

# Model
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(10, 64)  # Embedding layer (input size (tokens), embedding size)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(64, 8), 3)  #  (dimensionality of the model,  attention heads), num encoder layers)
        self.transformer_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(64, 8), 3)  #  (dimensionality of the model,  attention heads), num decoder layers)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(1, 0, 2)  # Reshape for Transformer input (sequence_length, batch_size, embedding_size) to (batch_size, sequence_length, embedding_size)

        encoded = self.transformer_encoder(embedded)
        decoded = self.transformer_decoder(embedded, encoded)


        decoded = decoded.permute(1, 0, 2)  # Reshape for linear layer input

        x = self.relu(self.fc1(decoded[:, -1, :])) # (batch_size, sequence_length(last element), embedding_size), Extracts the last time step representation for each sequence in the batch
        x = self.fc2(x)
        return x

model = Transformer().to(device)

# Data
input_data = torch.randint(0, 10, (10, 4)).to(device) # random number can only be between 0 and 10(max 9)  with (10,4) shape because embedding layer is 10 so only numbers between 0 and 9 is acceptable
#input_data = (torch.rand(10,4) * 10).to(device).long() # same thing

target = torch.rand(10, 4).to(device)
#target = torch.randint(0,10,(10, 4)).to(device) # it doesnt matter for the target

#print(target)
#print(input_data)

learning_rate = 0.001
num_epochs = 100

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(input_data)

    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch: [{epoch} / {num_epochs}], Loss: {loss:.5f}')

print(f'Epoch: [{epoch} / {num_epochs}], Loss: {loss:.5f}')
