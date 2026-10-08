import torch
from model import build_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

model = build_model()

state = torch.load(
    "checkpoints/cnn_cifar10.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state)

model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:
        
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct +=(predictions == labels).sum().item()
        total += labels.shape[0]

accuracy = correct / total

print("Model loaded successfully!")
print("Test accuracy:", accuracy)