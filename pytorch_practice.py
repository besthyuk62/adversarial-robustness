import torch
import torch.nn as nn

x = torch.tensor([
    [1.0,  2.0, -1.0],
    [0.0,  1.0,  2.0],
    [2.0, -1.0,  1.0],
    [1.0,  0.0,  1.0]
])

target = torch.tensor([
    [ 3.0, -1.0],
    [ 1.0,  2.0],
    [ 4.0,  0.0],
    [ 2.0,  1.0]
])

model = nn.Sequential(
    nn.Linear(3, 4),
    nn.ReLU(),
    nn.Linear(4,2)
)



optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(1000):

    optimizer.zero_grad()

    y = model(x)

    loss = ((y - target)**2).mean()

    loss.backward()

    optimizer.step()

    if step % 100 == 0:
        print(step, loss)

final = model(x)
print(final)
print(target)
