import torch

DEVICE = "cuda"
SIZE = 3072
ITERATIONS = 20

a = torch.randn((SIZE, SIZE), device=DEVICE)
b = torch.randn((SIZE, SIZE), device=DEVICE)
c = torch.randn((SIZE, SIZE), device=DEVICE)
d = torch.randn((SIZE, SIZE), device=DEVICE)

# Warmup
for _ in range(3):
    _ = a @ b
    _ = c @ d
torch.cuda.synchronize()

# ---------------------------------------------------------
# Single stream
# ---------------------------------------------------------
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()

for _ in range(ITERATIONS):
    x = a @ b
    y = c @ d

end.record()
torch.cuda.synchronize()

single_stream_ms = start.elapsed_time(end)

# ---------------------------------------------------------
# Two streams
# ---------------------------------------------------------
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()

for _ in range(ITERATIONS):

    with torch.cuda.stream(stream1):
        x = a @ b

    with torch.cuda.stream(stream2):
        y = c @ d

# Make default stream wait for both streams
torch.cuda.current_stream().wait_stream(stream1)
torch.cuda.current_stream().wait_stream(stream2)

end.record()
torch.cuda.synchronize()

multi_stream_ms = start.elapsed_time(end)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Single stream: {single_stream_ms:.2f} ms")
print(f"Two streams:   {multi_stream_ms:.2f} ms")
