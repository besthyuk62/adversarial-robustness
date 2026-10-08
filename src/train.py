import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim

from model import build_model

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

model = build_model()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

images, lables = next(iter(train_loader))

epochs = 5

for epoch in range(epochs):
    
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        
        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
    
    print(
        "Epoch",
        epoch + 1,
        "Average loss:",
        running_loss / len(train_loader)
    )

classes = [
    "airplan", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
]

class_correct = [0] * 10
class_total = [0] * 10

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        for label, prediction in zip(labels, predictions):
            class_total[label.item()] += 1

            if prediction.item() == label.item():
                class_correct[label.item()] += 1

for i in range(10):
    accuracy = class_correct[i] / class_total[i]
    print(classes[i], ":", accuracy)

torch.save(
    model.state_dict(),
    "checkpoints/cnn_cifar10.pth"
)
print("Model saved")