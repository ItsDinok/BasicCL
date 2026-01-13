"""
This is a basic toy CL setip designed to be modular and expandable
The goal of this is just to get the basics down so I can understand the key points and build on them
The next step is to create a robust model and smarter evaluation metrics
This is task incremental learning, where each class has an output layer, hence the no forgetting
"""
# TODO: Comment all of this

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

class CLMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_tasks=5, output_per_task=2):
        super().__init__()
        self.shared_fc = nn.Linear(input_size, hidden_size)
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_size, output_per_task) for _ in range(num_tasks)    
        ])

    def forward(self, x, task_id):
        x = x.view(x.size(0), -1)
        h = F.relu(self.shared_fc(x))
        out = self.task_heads[task_id](h)
        return out


def get_task_data(dataset, task_id, classes_per_task):
    start_class = task_id * classes_per_task
    end_class = start_class + classes_per_task
    idx = [i for i, (_, y) in enumerate(dataset) if start_class <= y < end_class]
    subset = Subset(dataset, idx)

    X_task = torch.stack([dataset[i][0] for i in idx])
    y_task = torch.tensor([dataset[i][1] - start_class for i in idx])
    return X_task, y_task


def evaluate_task(model, dataset, task_id, device, classes_per_task = 2):
    X_test, y_test = get_task_data(dataset, task_id, classes_per_task)
    X_test, y_test = X_test.to(device), y_test.to(device)
    model.eval()

    with torch.no_grad():
        output = model(X_test, task_id)
        preds = torch.argmax(output, dim = 1)
        acc = (preds == y_test).float().mean().item()
    model.train()
    return acc


def main():
    model = CLMLP(num_tasks = 3, output_per_task=2)
    optimiser = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    full_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    full_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_tasks = 3
    classes_per_task = 2
    model = CLMLP(num_tasks=num_tasks, output_per_task=classes_per_task).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(model.parameters(), lr = 0.01)

    for task_id in range(num_tasks):
        X_train, y_train = get_task_data(full_train, task_id, classes_per_task)
        X_train, y_train = X_train.to(device), y_train.to(device)

        # Simple mini-batch training
        batch_size = 64
        for epoch in range(3):
            perm = torch.randperm(X_train.size(0))
            for i in range(0, X_train.size(0), batch_size):
                idx = perm[i:i + batch_size]
                x_batch, y_batch = X_train[idx], y_train[idx]
                optimiser.zero_grad()
                output = model(x_batch, task_id)
                loss = criterion(output, y_batch)
                loss.backward()
                optimiser.step()

        for t in range(num_tasks):
            acc = evaluate_task(model, full_test, t, device, classes_per_task)
            print(f"Task {t} test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
