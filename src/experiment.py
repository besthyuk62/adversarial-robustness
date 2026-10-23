import torch
import torch.nn as nn
from model import build_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import statistics
import time

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

def evaluate_pgd(steps, restarts, max_batches):

    model.eval()

    clean_correct = 0
    standard_correct = 0
    standard_survived_num = 0
    survived_num = 0
    total_images = 0

    new_successes = [0] * restarts

    for batch_idx, (images, labels) in enumerate(test_loader):

        if batch_idx >= max_batches:
            break

        standard_adv = pgd_attack(
            images, labels,
            epsilon_norm, alpha_norm,
            steps=steps,
            random_start=False
        )

        with torch.no_grad():
            clean_preds = model(images).argmax(dim=1)
            clean_correct += (clean_preds == labels).sum().item()

            standard_preds = model(standard_adv).argmax(dim=1)
            standard_correct += (standard_preds == labels).sum().item()

            survived = (clean_preds == labels) & (standard_preds == labels)

            standard_survived_num += survived.sum().item()
        
        for restart in range(restarts):

            random_adv = pgd_attack(
                images, labels,
                epsilon_norm, alpha_norm,
                steps=steps,
                random_start=True
            )

            with torch.no_grad():
                random_preds = model(random_adv).argmax(dim=1)

                newly_broken = (
                    survived & (random_preds != labels)
                ).sum().item()

                new_successes[restart] += newly_broken

                survived = survived & (random_preds == labels)
        
        survived_num += survived.sum().item()
        total_images += labels.size(0)

    return {
        "clean_acc": clean_correct / total_images,
        "standard_acc": standard_correct / total_images,
        "standard_robust_acc": standard_survived_num / total_images,
        "multi_robust_acc": survived_num / total_images,
        "new_successes": new_successes
}

epsilon_norm = (2 / 255) / 0.5
alpha_norm = (0.5 / 255) / 0.5

seeds = [42, 123, 2026]

accuracies = []
times = []

for seed in seeds:

    torch.manual_seed(seed)

    start = time.perf_counter()

    result = evaluate_pgd(
        steps=10,
        restarts=3,
        max_batches=8
    )

    elapsed = time.perf_counter() - start

    accuracies.append(result["multi_robust_acc"])
    times.append(elapsed)

    print(
        f"Seed {seed}: "
        f"Accuracy = {result['multi_robust_acc']:.4f}, "
        f"Time = {elapsed:.3f}s, "
        f"New Successes = {sum(result['new_successes'])}"
    )

print("Mean Accuracy:", statistics.mean(accuracies))
print("Std Accuracy:", statistics.stdev(accuracies))

print("Mean Time:", statistics.mean(times))
print("Std Time:", statistics.stdev(times))