import torch
import torch.nn as nn

x = torch.randn(4, 3, 32, 32)

model = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(8, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Flatten(),

    nn.Linear(1024, 10)
)


target = torch.tensor([2, 5, 1, 7])

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

for step in range(200):

    optimizer.zero_grad()

    y = model(x)

    loss = criterion(y, target)

    loss.backward()

    optimizer.step()

    if step % 20 == 0:
        pred = y.argmax(dim=1)
        correct = (pred == target).sum()
        accuracy = correct / target.shape[0]

        print(
            "step:", step,
            "loss:", loss.item(),
            "accuracy:", accuracy.item()
        )
