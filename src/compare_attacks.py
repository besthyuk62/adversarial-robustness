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

epsilon = 4 / 255
alpha = 1 / 255
steps = 7

epsilon_norm = epsilon / 0.5
alpha_norm = alpha / 0.5

clean_correct = 0
fgsm_correct = 0
pgd_correct = 0
total_images = 0

for images, labels in test_loader:

    fgsm_images = images.clone()
    pgd_images = images.clone()

    # 2. FGSM 공격 이미지 생성 (Gradient 1회)
    fgsm_images = fgsm_images.detach().requires_grad_(True)
    outputs = model(fgsm_images)
    loss = criterion(outputs, labels)
    loss.backward()
    fgsm_images = fgsm_images + epsilon_norm * fgsm_images.grad.sign()
    fgsm_images = fgsm_images.clamp(-1, 1)

    # 3. PGD 공격 이미지 생성 (Gradient 7회)
    for step in range(steps):
        pgd_images = pgd_images.detach().requires_grad_(True)
        outputs = model(pgd_images)
        loss = criterion(outputs, labels)
        loss.backward()

        with torch.no_grad():

            pgd_images = (
                pgd_images + alpha_norm * pgd_images.grad.sign()
            )

            pgd_images = torch.minimum(images + epsilon_norm, pgd_images)
            pgd_images = torch.maximum(images - epsilon_norm, pgd_images)
            pgd_images = pgd_images.clamp(-1, 1)

    with torch.no_grad():

        # 4. Clean 예측
        clean_outputs = model(images)
        clean_preds = clean_outputs.argmax(dim=1)

        # 5. FGSM 예측
        fgsm_outputs = model(fgsm_images)
        fgsm_preds = fgsm_outputs.argmax(dim=1)

        # 6. PGD 예측
        pgd_outputs = model(pgd_images)
        pgd_preds = pgd_outputs.argmax(dim=1)

        # 7. 각 방법의 정답 개수 누적
        clean_correct += (clean_preds == labels).sum().item()
        fgsm_correct += (fgsm_preds == labels).sum().item()
        pgd_correct += (pgd_preds == labels).sum().item()

    # 8. 전체 이미지 개수 누적
    total_images += labels.size(0)

# 9. 세 가지 Accuracy 계산 및 출력
print(f"Clean accuracy: {clean_correct / total_images:.4f}")
print(f"FGSM accuracy: {fgsm_correct / total_images:.4f}")
print(f"PGD accuracy: {pgd_correct / total_images:.4f}")

print(
    "PGD max perturbation:",
    (pgd_images - images).abs().max().item()
)
print("Epsilon limit:", epsilon_norm)