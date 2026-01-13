import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import datasets, transforms, models
import random

# TODO: Write summary
# TODO: Rewrite modularly

class ReplayBuffer(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# Basic residual block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # First conv layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
            padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Second conv layer
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
            padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ResNet model
class ResNet(nn.Module):
    def __init__(self, block, blocks, classes=1000):
        super().__init__()
        self.in_channels = 64

        # Initial conv
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks[3], stride=2)

        # Final classidier
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        strides = [stride] + [1]*(blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# CIL ResNet model
class IncrementalResNet(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(512, 0)
        self.n_classes = 0

    def add_classes(self, n_new):
        old_weight = self.fc.weight.data if self.n_classes > 0 else None
        old_bias = self.fc.bias.data if self.n_classes > 0 else None

        new_fc = nn.Linear(self.fc.in_features, self.n_classes + n_new)

        if self.n_classes > 0:
            new_fc.weight.data[:self.n_classes] = old_weight
            new_fc.bias.data[:self.n_classes] = old_bias

        self.fc = new_fc
        self.n_classes += n_new

    def forward(self, x):
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        out = self.fc(features)
        return out


# Instantiator
def resnet18(classes = 1000):
    return ResNet(BasicBlock, [2,2,2,2], classes)


# Settings
classes_per_task = 2
memory_size = 2000 # Replay  buffer
batch_size = 32
lr = 0.001
tasks = 5 # 10 classes in total

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Example with CIFAR-10
full_dataset = datasets.CIFAR10(root = "./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root = "./data", train=False, download=True, transform=transform)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = models.resnet18(pretrained=False)
feature_extractor = nn.Sequential(
    resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
    resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
    resnet.avgpool
)
model = IncrementalResNet(feature_extractor)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=lr, momentum = 0.9)

# Replay memory
memory_data = []
memory_labels = []

# Incremental learning loop
for task in range(tasks):
    # Select classes for this task
    classes = list(range(task * classes_per_task, (task + 1) * classes_per_task))
    idx = [i for i, (_, label) in enumerate(full_dataset) if label in classes]
    task_dataset = Subset(full_dataset, idx)

    # Extend layer for new classes
    model.add_classes(len(classes))
    model = model.to(device)
    optmiser = optim.SGD(model.parameters(), lr = lr, momentum = 0.9)

    # Use replay buffer
    if memory_data:
        memory_dataset = ReplayBuffer(memory_data, memory_labels)
        train_loader = DataLoader(
            ConcatDataset([task_dataset, memory_dataset]),
            batch_size = batch_size,
            shuffle = True
        )
    else:
        train_loader = DataLoader(task_dataset, batch_size = batch_size, shuffle = True)


    # Training
    model.train()
    for epoch in range(5):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimiser.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

    # Update memory via random sampling
    for img, lbl in DataLoader(task_dataset, batch_size=1, shuffle=True):
        if len(memory_data) >= memory_size:
            break
        memory_data.append(img.squeeze(0))
        memory_labels.append(lbl.item())

    print(f"Finished training task: {task + 1} / {tasks}")

# Evaluation
model.eval()
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Accuracy on all classes: {correct / total:.4f}")
