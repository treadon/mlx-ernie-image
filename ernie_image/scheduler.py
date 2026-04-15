"""Flow Matching Euler Discrete Scheduler for ERNIE-Image."""

import mlx.core as mx


class FlowMatchEulerScheduler:
    """Simple flow matching scheduler with Euler stepping.

    Flow matching predicts velocity v, and the update is:
        x_{t-1} = x_t + dt * v

    Sigmas go from 1.0 (pure noise) to 0.0 (clean image).
    """

    def __init__(self, shift: float = 4.0, num_train_timesteps: int = 1000):
        self.shift = shift
        self.num_train_timesteps = num_train_timesteps
        self.sigmas = None
        self.timesteps = None

    def set_timesteps(self, num_inference_steps: int):
        """Create linearly spaced sigmas with shift applied."""
        sigmas = mx.linspace(1.0, 0.0, num_inference_steps + 1)

        # Apply shift: sigma_shifted = shift * sigma / (1 + (shift - 1) * sigma)
        shifted = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        self.sigmas = shifted
        self.timesteps = shifted[:-1] * self.num_train_timesteps

    def step(self, model_output, timestep_idx: int, sample):
        """Euler step: x_{t-1} = x_t + dt * v"""
        sigma = self.sigmas[timestep_idx]
        sigma_next = self.sigmas[timestep_idx + 1]
        dt = sigma_next - sigma

        prev_sample = sample + dt * model_output
        return prev_sample
