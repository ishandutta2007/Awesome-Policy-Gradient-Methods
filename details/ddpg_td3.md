# Deep Deterministic Policy Gradient (DDPG / TD3)

## Overview
**DDPG** and **TD3** are actor-critic variants designed for continuous action spaces, utilizing deterministic policies.

## Deterministic Optimization Loop
```mermaid
graph TD
    A[State s] --> B[Actor: Deterministic Action a = mu_theta]
    B --> C[Critic: Action-Value Q_phi]
    C --> D[Update Actor using Critic's Gradient w.r.t action]
```

[← Back to README](../README.md)