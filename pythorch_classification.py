import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.datasets import load_breast_cancer

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Create a dataloader to batch and shuffle the data
dataloader = DataLoader(list(zip(X, y)), batch_size=32, shuffle=True)

# Define the neural network


class BreastCancerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 100)
        self.fc6 = nn.Linear(100, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x


# Initialize the model and optimizer
model = BreastCancerNet()
optimizer = torch.optim.Adam(model.parameters())

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Initialize the TensorBoard writer
writer = SummaryWriter()

# Train the model
for epoch in range(100):
    for x, y in dataloader:
        # Forward pass
        y_pred = model(x)

        # Compute the loss
        loss = criterion(y_pred, y)

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Zero the gradients
        optimizer.zero_grad()

        # Log the loss to TensorBoard
        writer.add_scalar('Training Loss', loss.item(), global_step=epoch)


# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for x, y in dataloader:
        y_pred = model(x)
        _, predicted = torch.max(y_pred.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

# Print the accuracy
accuracy = 100 * correct / total
print('Accuracy: {}'.format(accuracy))

# Plot the results using matplotlib

# Get the predictions for the whole dataset
y_pred = model(X)
_, predicted = torch.max(y_pred.data, 1)

# Create a confusion matrix
confusion_matrix = torch.zeros(2, 2)
for i in range(len(y)):
    confusion_matrix[y[i], predicted[i]] += 1

# Plot the confusion matrix using matplotlib
plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks([0, 1])
plt.yticks([0, 1])

# Add the values to the plot
for i in range(2):
    for j in range(2):
        plt.text(j, i, confusion_matrix[i, j].item(
        ), ha='center', va='center', color='white')

plt.show()
