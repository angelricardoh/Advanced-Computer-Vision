import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # use GPU if available

class Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(3, 6, 5, stride=1)
      self.pool = nn.MaxPool2d(2, 2)
      self.conv2 = nn.Conv2d(6, 16, 5, stride=1)
      self.fc1 = nn.Linear(16 * 5 * 5, 120)
      self.fc2 = nn.Linear(120, 84)
      self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
      # (batch_size, 3, 32, 32)
      x = self.pool(F.relu(self.conv1(x))) # => (b, 6, 28, 28) => (b, 6, 14, 14)
      x = self.pool(F.relu(self.conv2(x))) # => (b, 16, 10, 10) => (b, 16, 5, 5)

      x = x.view(-1, 16 * 5 * 5) # => [b, 16 * 5 * 5]

      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))

      x = self.fc3(x)
      return x

net = Net()
net = net.to(device)