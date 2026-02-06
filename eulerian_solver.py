import numpy as np

def compute_velocity_field(density):
    # Simplified numerical computation of gradients (placeholder)
    grad_y, grad_x = np.gradient(density)
    # Use gradients to compute a basic “velocity” field
    velocity = np.sqrt(grad_x**2 + grad_y**2)
    return velocity

def apply_boundary_conditions(state):
    # Apply zero-boundary conditions
    state[0, :] = state[-1, :] = 0
    state[:, 0] = state[:, -1] = 0
    return state

def simulate_fluid(total_timesteps, grid_size=(100, 100)):
    # Initialize density field with a central spike
    density = np.zeros(grid_size)
    density[grid_size[0]//2, grid_size[1]//2] = 1.0

    simulation_data = []
    for t in range(total_timesteps):
        velocity = compute_velocity_field(density)
        density = apply_boundary_conditions(density)
        # Save the state as a dictionary (can be extended to store pressure, etc.)
        simulation_data.append({
            'density': density.copy(),
            'velocity': velocity.copy()
        })
        # Update density for the next step (this is oversimplified)
        density = density * 0.99 + np.random.randn(*grid_size)*0.001  # Add slight perturbation
    return simulation_data

if __name__ == '__main__':
    data = simulate_fluid(total_timesteps=50)
    np.save('simulation_data.npy', data)