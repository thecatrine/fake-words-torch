# -*- coding: utf-8 -*-
import torch
import math

dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0")

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.tan(x)

p = torch.tensor([1, 2, 3, 4, 5])
xx = x.unsqueeze(-1).pow(p)

model = torch.nn.Sequential(
    torch.nn.Linear(5, 1),
    torch.nn.Flatten(0, 1),
)

loss_fn = torch.nn.MSELoss(reduction="sum")

learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
for t in range(20000):
    y_pred = model(xx)

    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

linear_layer = model[0]

print(f"""Result: y = \
    {linear_layer.bias.item()} + \
    {linear_layer.weight[:, 0].item()} x + \
    {linear_layer.weight[:, 1].item()} x^2 + \
    {linear_layer.weight[:, 2].item()} x^3 + \
    {linear_layer.weight[:, 3].item()} x^4 + \
    {linear_layer.weight[:, 4].item()} x^5""")