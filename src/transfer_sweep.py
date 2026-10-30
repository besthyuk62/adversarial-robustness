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

def evaluate_transfer(epsilon_pixel, max_batches=8):

    epsilon_norm = (epsilon_pixel / 255) / 0.5
    alpha_norm = epsilon_norm / 4

    common_total = 0

    success_AA = 0
    success_AB = 0
    success_BB = 0
    success_BA = 0

    for batch_idx, (images, labels) in enumerate(test_loader):

        if batch_idx >= max_batches:
            break

        with torch.no_grad():
            clean_A_preds = model(images).argmax(dim=1)
            clean_B_preds = model_B(images).argmax(dim=1)

            both_correct = (
                (clean_A_preds == labels) &
                (clean_B_preds == labels)
            )

        common_total += both_correct.sum().item()

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

        with torch.no_grad():

            # TODO 3: adv_A를 A와 B에 각각 입력
            adv_AA_preds = model(adv_A).argmax(dim=1)
            adv_AB_preds = model_B(adv_A).argmax(dim=1)
            # TODO 4: adv_B를 B와 A에 각각 입력
            adv_BA_preds = model(adv_B).argmax(dim=1)
            adv_BB_preds = model_B(adv_B).argmax(dim=1)
            # TODO 5: 네 가지 공격의 성공 수 누적
            success_AA += (
                both_correct &
                (adv_AA_preds != labels)
                ).sum().item()

            success_AB += (
                both_correct &
                (adv_AB_preds != labels)
                ).sum().item()

            success_BB += (
                both_correct &
                (adv_BB_preds != labels)
                ).sum().item()

            success_BA += (
                both_correct &
                (adv_BA_preds != labels)
                ).sum().item()
    return {
        "AA": success_AA / common_total,
        "AB": success_AB / common_total,
        "BB": success_BB / common_total,
        "BA": success_BA / common_total
    }

epsilons = [0, 1, 2, 4, 8]

AA_results = []
AB_results = []
BB_results = []
BA_results = []

for eps in epsilons:
    result = evaluate_transfer(epsilon_pixel=eps, max_batches=8)

    AA_results.append(result["AA"])
    AB_results.append(result["AB"])
    BB_results.append(result["BB"])
    BA_results.append(result["BA"])

    print(f"Epsilon: {eps}/255")
    print(result)

fig, ax = plt.subplots(figsize=(8, 5))

curves = [
    ("A to A (White-box)", AA_results, "C0", "-", "o"),
    ("A to B (Transfer)", AB_results, "C0", "--", "s"),
    ("B to B (White-box)", BB_results, "C1", "-", "o"),
    ("B to A (Transfer)", BA_results, "C1", "--", "s")
]

for label, values, color, linestyle, marker in curves:
    ax.plot(
        epsilons,
        [v * 100 for v in values],
        color=color,
        linestyle=linestyle,
        marker=marker,
        linewidth=2,
        markersize=6,
        label=label
    )

ax.set_xlabel("Perturbation Budget ε (1/255)")
ax.set_ylabel("Attack Success Rate (%)")
ax.set_title("White-box vs. Transfer Attack Success Rate")

ax.set_xticks(epsilons)
ax.set_ylim(0, 105)

ax.grid(True, alpha=0.3)
ax.legend(loc="upper left")

fig.tight_layout()
fig.savefig(
    "transferability_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
