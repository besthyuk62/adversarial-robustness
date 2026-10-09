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

images, labels = next(iter(test_loader))

images.requires_grad_(True)

outputs = model(images)

criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)

loss.backward()

epsilon = 8 / 255
epsilon_norm = epsilon / 0.5

perturbation = epsilon_norm * images.grad.sign()

adv_images = images + perturbation
adv_images = adv_images.clamp(-1, 1).detach()



print("Original shape:", images.shape)
print("Adversarial shape:", adv_images.shape)

diff = (adv_images - images).abs()

print("Maximum perturbation:", diff.max().item())
print("Epsilon limit:", epsilon_norm)

with torch.no_grad():

    clean_outputs = model(images)
    adv_outputs = model(adv_images)

    clean_preds = clean_outputs.argmax(dim=1)
    adv_preds = adv_outputs.argmax(dim=1)

    clean_accuracy = (clean_preds == labels).float().mean()
    adv_accuracy = (adv_preds == labels).float().mean()


print("Clean accuracy:", clean_accuracy.item())
print("FGSM accuracy:", adv_accuracy.item())

import matplotlib.pyplot as plt

idx = 0

original = (images[idx].detach().permute(1, 2, 0) + 1) / 2

attacked = (adv_images[idx].permute(1, 2, 0) + 1) / 2

difference = attacked - original
difference_vis = (difference / (2 * epsilon) + 0.5).clamp(0, 1)

plt.figure(figsize=(10, 3))

plt.subplot(1, 3, 1)
plt.imshow(original)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(attacked)
plt.title("FGSM")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(difference_vis)
plt.title("Perturbation (scaled)")
plt.axis("off")

plt.show()

random_accuracies = []

for trial in range(30):
    random_sign = (torch.rand_like(images) >= 0.5).float() * 2 - 1
    
    random_perturbation = epsilon_norm * random_sign

    random_images = images + random_perturbation
    random_images = random_images.clamp(-1, 1).detach()

    with torch.no_grad():
        random_outputs = model(random_images)

        random_preds = random_outputs.argmax(dim=1)

        random_accuracy = (random_preds == labels).float().mean()

    random_accuracies.append(random_accuracy)

results = torch.tensor(random_accuracies)

print("Random Mean:", results.mean().item())
print("Random Std:", results.std(unbiased=False).item())
print("Random Min:", results.min().item())
print("Random Max:", results.max().item())