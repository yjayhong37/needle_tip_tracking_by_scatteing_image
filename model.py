import torch
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 200 * 200, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

"""

added an attention layer to see if it works better


# Define the attention layer
class AttentionLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = torch.matmul(q, k.transpose(2, 1))  # Compute attention scores
        scores = self.softmax(scores)  # Apply softmax to normalize scores

        attended = torch.matmul(scores, v)  # Apply attention to the value vectors

        return attended


# Modify the MyModel class to include the attention layer
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.attention = AttentionLayer(16 * 200 * 200, 16 * 200 * 200)  # Add attention layer
        self.fc = nn.Sequential(
            nn.Linear(16 * 200 * 200, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.attention(x)  # Apply attention layer
        x = self.fc(x)
        return x
        
        """
