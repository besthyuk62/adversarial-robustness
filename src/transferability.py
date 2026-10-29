import torch
import torch.nn as nn
from model import build_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

model = build_model()

state = torch.load(
    "checkpoints/cnn_cifar10.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state)

model.eval()
model.requires_grad_(False)

model_B = build_model()

state_B = torch.load(
    "checkpoints/cnn_cifar10_seed123.pth",
    map_location="cpu",
    weights_only=True
)

model_B.load_state_dict(state_B)

model_B.eval()
model_B.requires_grad_(False)

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
criterion_sum = nn.CrossEntropyLoss(reduction="sum")

def pgd_attack(
    attack_model, images, labels, epsilon_norm, alpha_norm, 
    steps, record_steps=None, random_start=False
    ):

    original_images = images.detach()
    if random_start:
        random_noise = torch.rand_like(original_images) * 2 - 1
        random_noise = random_noise * epsilon_norm
        adv_images = original_images + random_noise
        adv_images = adv_images.clamp(-1, 1)
    else:    
        adv_images = original_images.clone()

    saved = {}
    
    for step in range(1, steps + 1):
        adv_images = adv_images.detach().requires_grad_(True)
        outputs = attack_model(adv_images)
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

images, labels = next(iter(test_loader))

epsilon_norm = (2 / 255) / 0.5
alpha_norm = (0.5 / 255) / 0.5

total_images = 0

clean_A_correct = 0
clean_B_correct = 0

survived_AA = 0
survived_AB = 0
survived_BB = 0
survived_BA = 0

for batch_idx, (images, labels) in enumerate(test_loader):

    if batch_idx >= 8:
        break

    # Model A에서 PGD 공격 생성
    adv_A = pgd_attack(
        model,
        images, labels,
        epsilon_norm, alpha_norm,
        steps=10,
        random_start=False
    )

    adv_B = pgd_attack(
        model_B,
        images, labels,
        epsilon_norm, alpha_norm,
        steps=10,
        random_start=False
    )
    # Clean A, Clean B 예측
    with torch.no_grad():

        clean_A_preds = model(images).argmax(dim=1)
        clean_B_preds = model_B(images).argmax(dim=1)

        adv_A_preds = model(adv_A).argmax(dim=1)
        adv_B_preds = model_B(adv_A).argmax(dim=1)
        adv_BB_preds = model_B(adv_B).argmax(dim=1)
        adv_BA_preds = model(adv_B).argmax(dim=1)
 
    # 각 정답 개수 누적
    clean_A_correct += (clean_A_preds == labels).sum().item()
    clean_B_correct += (clean_B_preds == labels).sum().item()

    survived_AB += (
        (clean_B_preds == labels) &
        (adv_B_preds == labels)
    ).sum().item()

    survived_AA += (
        (clean_A_preds == labels) &
        (adv_A_preds == labels)
    ).sum().item()

    survived_BB += (
        (clean_B_preds == labels) &
        (adv_BB_preds == labels)
    ).sum().item()

    survived_BA += (
        (clean_A_preds == labels) &
        (adv_BA_preds == labels)
    ).sum().item()


    total_images += labels.size(0)

transfer_asr = (
    1 - survived_AB / clean_B_correct
    if clean_B_correct > 0
    else float("nan")
)

whitebox_asr = (
    1 - survived_AA / clean_A_correct
    if clean_A_correct > 0
    else float("nan")
)

whitebox_BB_asr = (
    1 - survived_BB / clean_B_correct
    if clean_B_correct > 0
    else float("nan")
)

transfer_BA_asr = (
    1 - survived_BA / clean_A_correct
    if clean_A_correct > 0
    else float("nan")
)

print("Model A Clean Accuracy:", clean_A_correct / total_images)
print("Model B Clean Accuracy:", clean_B_correct / total_images)
print("A → A White-box ASR:", whitebox_asr)
print("A → B Transfer Attack Success Rate:", transfer_asr)
print("B → B White-box ASR:", whitebox_BB_asr)
print("B → A Transfer Attack Success Rate:", transfer_BA_asr)
