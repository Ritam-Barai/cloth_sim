import torch
import torch.nn as nn

# Define num_keypoints (number of keypoints)
num_keypoints = 40

# Create the nn.Sequential model
model = nn.Sequential(
    nn.Linear(num_keypoints * 2, 40 * 2),  # *2 for x,y coordinates
    nn.ReLU(),
    # Lambda layer to scale outputs by 256
    nn.Lambda(lambda x: x * 256)
)

# Example input tensor
input_tensor = torch.randn(1, num_keypoints * 2)

# Forward pass through the model
output = model(input_tensor)
print(input_tensor,output)

