import os

details_dir = r"C:\Users\ishan\Documents\Projects\Awesome-Policy-Gradient-Methods\details"
os.makedirs(details_dir, exist_ok=True)

pages = {
    "trpo_era.md": """# The Statistical Distance & Trust Region Era (TRPO)

## Overview
**Trust Region Policy Optimization (TRPO)** (Schulman et al., 2015) introduced statistical distance bounds to stabilize policy updates. Instead of changing policy weights blindly, TRPO constrains the update using Kullback-Leibler (KL) divergence on the policy's action distribution output.

## Optimization Flow
```mermaid
graph TD
    A[Current Policy parameters] --> B[Compute Advantage Estimates]
    B --> C[Set KL Divergence constraint]
    C --> D[Solve Constrained Optimization via Conjugate Gradient]
    D --> E[Update Policy parameters within Trust Region]
```

## Key Characteristics
- **Second-Order Optimization:** Uses Fisher Information Matrix for constraint.
- **Monotonic Improvement:** Guarantees policy improvement mathematically.
- **High Compute Overhead:** Requires expensive matrix inversions.

[← Back to README](../README.md)""",

    "ppo_era.md": """# The Clipped Bound Proximal Era (PPO)

## Overview
**Proximal Policy Optimization (PPO)** (Schulman et al., 2017) simplified the complex trust region constraint of TRPO into a first-order optimization objective with ratio clipping.

## Clipped Objective Flow
```mermaid
graph LR
    A[Compute Ratio r_t] --> B{Is r_t outside clip window?}
    B -- Yes --> C[Clip Ratio to 1-epsilon or 1+epsilon]
    B -- No --> D[Use original Ratio]
    C --> E[Compute Pessimistic Objective Min]
    D --> E
```

## Key Characteristics
- **First-Order:** Uses standard gradient descent.
- **Ratio Clipping:** Clamps the probability ratio to prevent destructive updates.
- **State-of-the-Art:** default choice for general RL benchmarks.

[← Back to README](../README.md)""",

    "reinforce_variant.md": """# REINFORCE (Vanilla Policy Gradient)

## Overview
**REINFORCE** scales policy gradient directions by the raw cumulative return ($G_t$). 

## Optimization Formula
$$g = \\mathbb{E} \\left[ \\nabla_\\theta \\log \\pi_\\theta(a_t|s_t) G_t \\right]$$

## Process Flow
```mermaid
graph TD
    A[Generate Episode] --> B[Calculate Returns G_t]
    B --> C[Compute Log Probabilities]
    C --> D[Apply Gradient Step scaled by G_t]
```

[← Back to README](../README.md)""",

    "actor_critic.md": """# Advantage Actor-Critic (A2C / A3C)

## Overview
**Actor-Critic** methods reduce variance by introducing an estimator for expected returns (Critic) to baseline the Actor's policy updates.

## Actor-Critic Interaction
```mermaid
graph LR
    Actor[Actor: Policy pi] -- Action --> Env[Environment]
    Env -- State & Reward --> Critic[Critic: Value V]
    Critic -- Advantage A = Q - V --> Actor
```

[← Back to README](../README.md)""",

    "ddpg_td3.md": """# Deep Deterministic Policy Gradient (DDPG / TD3)

## Overview
**DDPG** and **TD3** are actor-critic variants designed for continuous action spaces, utilizing deterministic policies.

## Deterministic Optimization Loop
```mermaid
graph TD
    A[State s] --> B[Actor: Deterministic Action a = mu_theta]
    B --> C[Critic: Action-Value Q_phi]
    C --> D[Update Actor using Critic's Gradient w.r.t action]
```

[← Back to README](../README.md)""",

    "clipped_ppo.md": """# Clipped PPO Objective

## Overview
The clipped objective in PPO prevents the current policy from diverging too far from the old policy.

## Mathematical Objective
$$\\mathcal{L}_{\\text{CLIP}}(\\theta) = \\hat{\\mathbb{E}}_t \\left[ \\min(r_t(\\theta)\\hat{A}_t, \\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon)\\hat{A}_t) \\right]$$

## Clipping Boundaries
```mermaid
graph TD
    A[Policy Ratio r_t] --> B{Advantage > 0?}
    B -- Yes --> C[Limit upper bound to 1+epsilon]
    B -- No --> D[Limit lower bound to 1-epsilon]
```

[← Back to README](../README.md)""",

    "on_policy.md": """# On-Policy Policy Gradients

## Overview
On-policy methods strictly optimize parameters using samples generated directly by the current policy.

## Synchronization Loop
```mermaid
graph LR
    Policy[Current Policy] --> Rollout[Collect Trajectories]
    Rollout --> Update[Gradient Update]
    Update --> Discard[Discard Data]
    Discard --> Policy
```

[← Back to README](../README.md)""",

    "off_policy.md": """# Off-Policy Actor-Critic Adaptations

## Overview
Off-policy adaptations use experience replay buffers to recycle transitions gathered by historical policies.

## Replay Architecture
```mermaid
graph LR
    Env[Environment] --> Replay[Replay Buffer]
    Replay -- Sample Batches --> Policy[Target Policy Update]
    Policy --> ImportanceRatio[Importance Sampling Correction]
```

[← Back to README](../README.md)""",

    "coadaptation_desync.md": """# The Policy-Critic Co-Adaptation Desynchronization

## Overview
In Actor-Critic models, desynchronization occurs when the Critic updates too fast or slow relative to the Actor, leading to divergence.

## Polyak Averaging Mitigation
```mermaid
graph TD
    A[Critic Loss] --> B[Update Online Critic Network]
    B --> C[Soft-Update Target Critic via Polyak Averaging]
    Note over C: theta_target <- tau * theta_online + (1-tau) * theta_target
```

[← Back to README](../README.md)""",

    "sync_stalls.md": """# The Synchronous Mini-Batch Communication Stalls

## Overview
Distributed on-policy training suffers from CPU-to-GPU data transfer bottlenecks during environments rollout gathering.

## GPU Fused Physics Solution
```mermaid
graph LR
    GPUPhys[GPU-based Simulator] -- Direct Memory Access --> GPUBuffers[Tensor Buffers]
    GPUBuffers --> GPULearning[GPU Policy Optimization]
    Note over GPUPhys, GPULearning: Zero CPU-GPU round-trips
```

[← Back to README](../README.md)""",

    "post_training_alignment.md": """# Post-Training RL Alignment for Reasoning Models

## Overview
Applies policy gradient algorithms (like Clipped PPO) to align Large Language Models for structured reasoning.

## Alignment Pipeline
```mermaid
graph TD
    Prompt[Input Prompt] --> Policy[LLM Policy]
    Policy --> Rollout[Generates Tokens/Thought Trace]
    Rollout --> Sandbox[Verifiable Reward Environment / Code Sandbox]
    Sandbox --> Reward[Compute Reward Value]
    Reward --> Optimization[PPO Policy Gradient Step]
```

[← Back to README](../README.md)""",

    "torque_control.md": """# Continuous Kinetic Torque Control for Humanoid Robotics

## Overview
Coordinates high-frequency action loops for bipedal and quadrupedal robots using stochastic and deterministic policy gradients.

## Control Loop
```mermaid
graph LR
    Sensors[IMU & Force Sensors] --> EdgeTPU[Local Policy: SAC / TD3]
    EdgeTPU --> Torques[Kinetic Torque Outputs]
    Torques --> Physics[Robot Kinematics]
    Physics --> Sensors
```

[← Back to README](../README.md)""",

    "asset_trading.md": """# High-Frequency Multi-Agent Autonomous Asset Trading Matrices

## Overview
Executes quantitative investment allocations across financial landscapes using policy gradients.

## Trading Loop
```mermaid
graph TD
    Market[Market State Inputs] --> Agent[Policy Network]
    Agent --> Portfolio[Continuous Portfolio Weights]
    Portfolio --> Execution[Trade Execution Engine]
    Execution --> Cost[Transaction Cost Model]
    Cost --> Return[Reward Optimization Objective]
```

[← Back to README](../README.md)"""
}

for filename, content in pages.items():
    path = os.path.join(details_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
print("All pages written successfully.")
