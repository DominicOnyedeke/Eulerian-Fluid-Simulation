import torch
import numpy as np
import matplotlib.pyplot as plt
from model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained UNet model
model = UNet().to(device)
model.load_state_dict(torch.load('unet_model.pth', map_location=device))
model.eval()

def predict_next_frame(current_state):
    # Assume current_state is a NumPy array (H, W)
    input_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(input_tensor)
    return prediction.cpu().squeeze().numpy()

# Load a sample simulation state from the dataset
data = np.load('simulation_data.npy', allow_pickle=True)
sample_state = data[10]['density']

# Predict the next frame using the CNN
predicted_state = predict_next_frame(sample_state)

# Visualize the current state vs. the predicted state
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Current State")
plt.imshow(sample_state, cmap='viridis')
plt.subplot(1, 2, 2)
plt.title("Predicted Next State")
plt.imshow(predicted_state, cmap='viridis')
plt.show()