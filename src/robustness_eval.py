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

epsilons = [0, 2, 4, 8, 16]
fgsm_accuracies = []

for eps in epsilons:
    epsilon_norm = (eps / 255) / 0.5

    clean_correct = 0
    fgsm_correct = 0
    total_images = 0
    both_correct = 0

    for images, labels in test_loader:
        # Todo 1
        images.requires_grad_(True)

        #Todo 2
        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        #Todo 3
        perturbation = epsilon_norm * images.grad.sign()

        fgsm_images = images + perturbation
        fgsm_images = fgsm_images.clamp(-1, 1).detach()

        with torch.no_grad():
            #Todo 4
            clean_outputs = model(images)
            clean_preds = clean_outputs.argmax(dim=1)

            #Todo 5
            fgsm_outputs = model(fgsm_images)
            fgsm_preds = fgsm_outputs.argmax(dim=1)

            #Todo 6
            clean_correct += (clean_preds == labels).sum().item()
            fgsm_correct += (fgsm_preds == labels).sum().item()

            both_correct += ((clean_preds == labels) & (fgsm_preds == labels)).sum().item()
        
        #Todo 7
        total_images += labels.size(0)

    clean_accuracy = clean_correct / total_images
    fgsm_accuracy = fgsm_correct / total_images
    asr = 1 - both_correct / clean_correct
    fgsm_accuracies.append(fgsm_accuracy * 100)

    print("Epsilon", eps, "/ 255")
    print("clean Accuracy:", clean_accuracy)
    print("FGSM Accuracy:", fgsm_accuracy)
    print("Attack Success Rate:", asr)


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

# TODO 1. epsilons와 fgsm_accuracies를 선 그래프로 표현
plt.plot(epsilons, fgsm_accuracies, marker="o")
# TODO 2. X축 이름: "Epsilon (/255)"
plt.xlabel("Epsilon")
# TODO 3. Y축 이름: "Accuracy (%)"
plt.ylabel("Accuracy (%)")
# TODO 4. 그래프 제목: "FGSM Robustness Evaluation"
plt.title("FGSM Robustness Evaluation")

plt.xticks(epsilons)
plt.ylim(0, 70)
plt.grid(True)

plt.savefig("fgsm_robustness.png", dpi=200)
plt.show()