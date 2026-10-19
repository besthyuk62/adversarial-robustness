import torch
import torch.nn as nn
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
model.requires_grad_(False)

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

criterion = nn.CrossEntropyLoss()

def pgd_attack(
    images, labels, epsilon_norm, alpha_norm, 
    steps, record_steps=None
    ):

    original_images = images.detach()
    adv_images = original_images.clone()
    saved = {}
    
    for step in range(1, steps + 1):
        adv_images = adv_images.detach().requires_grad_(True)
        outputs = model(adv_images)
        loss = criterion(outputs, labels)
        loss.backward()

        with torch.no_grad():

            adv_images = (adv_images + alpha_norm * adv_images.grad.sign())

            adv_images = torch.minimum(original_images + epsilon_norm, adv_images)
            adv_images = torch.maximum(original_images - epsilon_norm, adv_images)
            adv_images = adv_images.clamp(-1, 1)
        
        if record_steps is not None and step in record_steps:
            saved[step] = adv_images.detach().clone()
    
    if record_steps is not None:
        return saved

    return adv_images.detach()


epsilon_norm = (2 / 255) / 0.5
alpha_norm = (0.5 / 255) / 0.5

correct_at_4 = 0
correct_at_20 = 0
total_images = 0


for images, labels in test_loader:

    adv_images = pgd_attack(
        images, labels, epsilon_norm, alpha_norm, steps=20, record_steps = [4, 20]
    )

    with torch.no_grad():
        outputs4 = model(adv_images[4])
        preds4 = outputs4.argmax(dim=1)
        correct_at_4 += (preds4 == labels).sum().item()
        
        outputs20 = model(adv_images[20])
        preds20 = outputs20.argmax(dim=1)
        correct_at_20 += (preds20 == labels).sum().item()

    total_images += labels.size(0)

print("Step 4 Accuracy:", correct_at_4 / total_images)
print("Step 20 Accuracy:", correct_at_20 / total_images)
