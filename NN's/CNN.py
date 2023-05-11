import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda")

# Dummy data generator
num_images = 1000
image_size = (32, 32)
num_channels = 3
images = np.random.rand(num_images, *image_size, num_channels)


# Random labels for the images
labels = np.random.randint(0, 2, num_images)


# Shows some sample images with their labels
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(images[i])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Image {i + 1}, Label {labels[i]}")
plt.tight_layout()
plt.show()


# Transpose the numpy array to have channels as the first dimension
images = np.random.rand(num_images, num_channels, *image_size)

# Converts the numpy arrays to PyTorch tensors
images_tensor = torch.tensor(images, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.int64)

# Creates the dataset
dataset = data.TensorDataset(images_tensor, labels_tensor)

# Loads the data
batch_size = 100
loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.convl1 = nn.Conv2d(num_channels, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.convl2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 64)  # output size = (input size - filter size + 2*padding) / 1 + 1 (don't do what I did and don't forget the order of operations)
        self.ReLu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(self.ReLu(self.convl1(x)))
        x = self.pool(self.ReLu(self.convl2(x)))
        x = x.view(-1, 32 * 5 * 5)  # flats the output to 1D tensor
        x = self.ReLu(self.fc1(x))
        x = self.fc2(x)

        return x


model = CNN().to(device)

learnnig_rate = 0.01
num_epochs = 10

Criterion = nn.CrossEntropyLoss()
Optimizer = torch.optim.SGD(model.parameters(), lr=learnnig_rate)

n_total_steps = len(loader)

# Train
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = Criterion(outputs, labels)

        Optimizer.zero_grad()
        Optimizer.step()
        loss.backward()

        if (i + 1) % 1 == 0:
            print(f'Epoch: [{epoch + 1}/{num_epochs}], Step: [{i + 1}/{n_total_steps}], Loss: {loss:.4f}')
