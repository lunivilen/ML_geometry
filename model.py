import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FCCube(nn.Module):
    def __init__(self):
        super(FCCube, self).__init__()
        # Полносвязные слои
        self.fc1 = nn.Linear(3, 16, device=device)
        self.fc2 = nn.Linear(16, 16, device=device)
        self.fc3 = nn.Linear(16, 8, device=device)
        self.fc4 = nn.Linear(8, 2, device=device)

        # Функция активации
        self.relu = nn.PReLU(num_parameters=1, init=0.25).to(device)

    def forward(self, x):
        # Прямой проход через полносвязные слои
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        return x
