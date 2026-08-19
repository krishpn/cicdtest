import os
import torch
import torch.nn as nn

# 1. Define a simple model (y = x * 2)
class SimpleModel(nn.Module):
    def forward(self, x):
        return x * 2.0

model = SimpleModel()
model.eval()
dummy_input = torch.randn(1, 4, dtype=torch.float32)

# Ensure directory exists
os.makedirs("model_repository/simple/1", exist_ok=True)

# 2. Export the ONNX model
torch.onnx.export(
    model,
    dummy_input,
    "model_repository/simple/1/model.onnx",
    input_names=["INPUT0"],
    output_names=["OUTPUT0"],
    dynamic_axes={"INPUT0": {0: "batch_size"}, "OUTPUT0": {0: "batch_size"}}
)

print("ONNX export complete.")