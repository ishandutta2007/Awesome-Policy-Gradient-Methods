# The Baseline Monte Carlo Era (REINFORCE)

## Overview
The **REINFORCE** algorithm (Williams, 1992) is the foundational Monte Carlo policy gradient method. It established that we can estimate the gradient of expected returns directly from trajectory rollouts without knowing the transition dynamics of the environment.

## Architecture & Flow
```mermaid
sequenceDiagram
    participant Agent
    participant Environment
    Agent->>Environment: Take action a_t ~ pi(a|s)
    Environment->>Agent: State s_{t+1}, Reward r_{t+1}
    Note over Agent: Complete Episode Trajectory
    Note over Agent: Compute Return G_t = sum(gamma^k * r_{t+k})
    Note over Agent: Update parameters: theta <- theta + alpha * grad(log pi(a_t|s_t)) * G_t
```

## Key Characteristics
- **On-Policy:** Requires samples generated from the current policy parameters.
- **High Variance:** Updates are based on full Monte Carlo returns ($G_t$), making them sensitive to noisy trajectories.
- **Unbiased:** The gradient estimate is mathematically unbiased.

[← Back to README](../README.md)
