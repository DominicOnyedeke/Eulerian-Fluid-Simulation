import numpy as np
import torch
from torch.utils.data import Dataset

class FluidSimulationDataset(Dataset):
    def __init__(self, data_file):
        # Load simulation data (a list of dicts)
        self.data = np.load(data_file, allow_pickle=True)

    def __len__(self):
        return len(self.data) - 1  # Because we form pairs [current, next]

    def __getitem__(self, idx):
        # Use density channel for this example (can be expanded to include velocity)
        current_state = self.data[idx]['density']
        next_state = self.data[idx+1]['density']
        # Normalize and add channel dimensions (1, H, W)
        current_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)
        next_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        return current_tensor, next_tensor

if __name__ == '__main__':
    dataset = FluidSimulationDataset('simulation_data.npy')
    print("Total samples:", len(dataset))