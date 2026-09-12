import torch

from src.models.lstm import LSTMForecaster


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = LSTMForecaster(
    input_size=1,
    hidden_size=64,
    num_layers=2,
    dropout=0.1,
).to(device)

x = torch.randn(
    64,
    24,
    1,
    device=device,
)

with torch.no_grad():
    y = model(x)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

trainable_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Device:", device)
print("Input:", x.shape)
print("Output:", y.shape)
print("Parameters:", parameter_count)
print("Trainable:", trainable_count)
print()
print(model)
