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


def evaluate_epsilon(epsilon_pixel, max_batches):

    epsilon_norm = (epsilon_pixel / 255) / 0.5

    clean_correct = 0
    fgsm_correct = 0
    pgd_correct = 0

    total_images = 0

    for batch_idx, (images, labels) in enumerate(test_loader):

        if batch_idx >= max_batches:
            break

        fgsm_adv = pgd_attack(
            model, 
            images, labels,
            epsilon_norm,
            epsilon_norm,
            steps=1,
            random_start=False
        )

        pgd_adv = pgd_attack(
            model, 
            images, labels,
            epsilon_norm,
            epsilon_norm / 4,
            steps=10,
            random_start=False
        )

        with torch.no_grad():
            clean_preds = model(images).argmax(dim=1)
            clean_correct += (clean_preds == labels).sum().item()

            fgsm_preds = model(fgsm_adv).argmax(dim=1)
            fgsm_correct += (
                (clean_preds == labels)&(fgsm_preds == labels)
                ).sum().item()

            pgd_preds = model(pgd_adv).argmax(dim=1)
            pgd_correct += (
                (clean_preds == labels)&(pgd_preds == labels)
                ).sum().item()

        total_images += labels.size(0)

    return {
        "epsilon": epsilon_pixel,
        "clean_acc": clean_correct / total_images,
        "fgsm_acc": fgsm_correct / total_images,
        "pgd_acc": pgd_correct / total_images,
}

def compare_attack_loss(epsilon_pixel, max_batches):
    epsilon_norm = (epsilon_pixel / 255) / 0.5

    clean_loss = 0
    fgsm_loss = 0
    pgd_loss = 0

    total_images = 0

    for batch_idx, (images, labels) in enumerate(test_loader):

        if batch_idx >= max_batches:
            break

        fgsm_adv = pgd_attack(
            model,
            images, labels,
            epsilon_norm,
            epsilon_norm,
            steps=1,
            random_start=False
        )

        pgd_adv = pgd_attack(
            model,
            images, labels,
            epsilon_norm,
            epsilon_norm / 4,
            steps=10,
            random_start=False
        )

        with torch.no_grad():
            clean_loss += criterion_sum(
                model(images), labels
            ).item()

            fgsm_loss += criterion_sum(
                model(fgsm_adv), labels
            ).item()

            pgd_loss += criterion_sum(
                model(pgd_adv), labels
            ).item()

        total_images += labels.size(0)

    return {
        "mean_clean_loss": clean_loss / total_images,
        "mean_fgsm_loss": fgsm_loss / total_images,
        "mean_pgd_loss": pgd_loss / total_images,
}

epsilons = [0, 1, 2, 4, 8]

results = []

for eps in epsilons:

    result1 = evaluate_epsilon(
        epsilon_pixel=eps,
        max_batches=8
    )

    result2 = compare_attack_loss(
        epsilon_pixel=eps,
        max_batches=8
    )
  
    combined_result = {**result1, **result2}
    results.append(combined_result)
    
    print(
        f"Epsilon = {eps}/255, "
        f"Clean = {result1['clean_acc']:.4f}, "
        f"FGSM = {result1['fgsm_acc']:.4f}, "
        f"PGD = {result1['pgd_acc']:.4f}, "
        f"Clean_loss = {result2['mean_clean_loss']:.4f}, "
        f"FGSM_loss = {result2['mean_fgsm_loss']:.4f}, "
        f"PGD_loss = {result2['mean_pgd_loss']:.4f}"
    )


eps_values = [r["epsilon"] for r in results]

fgsm_accs = [r["fgsm_acc"] * 100 for r in results]

pgd_accs = [r["pgd_acc"] * 100 for r in results]

plt.figure(figsize=(8, 5))

plt.plot(
    eps_values,
    fgsm_accs,
    marker="o",
    label="FGSM"
)

plt.plot(
    eps_values,
    pgd_accs,
    marker="s",
    label="PGD"
)

plt.xlabel("Epsilon (1/255)")
plt.ylabel("Robust Accuracy (%)")

plt.title("Adversarial Robustness: FGSM vs PGD")

plt.xticks(eps_values)
plt.ylim(0, 70)

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(
    "robustness_curve.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()