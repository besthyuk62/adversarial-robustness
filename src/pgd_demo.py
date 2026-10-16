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

criterion = nn.CrossEntropyLoss()
criterion_each = nn.CrossEntropyLoss(reduction="none")

epsilon = 4 / 255
alpha = 2 / 255
steps = 7

epsilon_norm = epsilon / 0.5
alpha_norm = alpha / 0.5

images, labels = next(iter(test_loader))

original_images = images.detach()
adv_images = original_images.clone()

for step in range(steps):

    adv_images = adv_images.detach().requires_grad_(True)
    outputs = model(adv_images)
    loss = criterion(outputs, labels)
    before_each = criterion_each(outputs, labels).detach()
    loss.backward()

    with torch.no_grad():

        adv_images = (
            adv_images + alpha_norm * adv_images.grad.sign()
        )

        adv_images = torch.minimum(original_images + epsilon_norm, adv_images)
        adv_images = torch.maximum(original_images - epsilon_norm, adv_images)
        adv_images = adv_images.clamp(-1, 1)

        loss_after = criterion(model(adv_images), labels)

        after_each = criterion_each(model(adv_images), labels)

        decreased = (after_each < before_each).sum().item()
    
    print(
        f"Step {step + 1}: "
        f"{loss.item():.4f} → {loss_after.item():.4f}"
    )

    print(f"Loss decreased in {decreased} images")


with torch.no_grad():
    clean_outputs = model(images)
    clean_preds = clean_outputs.argmax(dim=1)

    pgd_outputs = model(adv_images)
    pgd_preds = pgd_outputs.argmax(dim=1)

    clean_accuracy = (clean_preds == labels).float().mean()
    pgd_accuracy = (pgd_preds == labels).float().mean()


print("Clean accuracy:", clean_accuracy.item())

diff = (adv_images - original_images).abs()

# FGSM용 입력 준비
fgsm_input = original_images.clone().detach().requires_grad_(True)

# Gradient 계산
fgsm_outputs = model(fgsm_input)
fgsm_loss = criterion(fgsm_outputs, labels)
fgsm_loss.backward()

# FGSM 공격 이미지 생성
fgsm_images = (
    fgsm_input.detach()
    + epsilon_norm * fgsm_input.grad.sign()
).clamp(-1, 1)

# 정확도 계산
with torch.no_grad():
    fgsm_preds = model(fgsm_images).argmax(dim=1)
    fgsm_accuracy = (fgsm_preds == labels).float().mean()

print("FGSM Accuracy:", fgsm_accuracy.item())
print("PGD Accuracy:", pgd_accuracy.item())