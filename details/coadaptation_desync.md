# The Policy-Critic Co-Adaptation Desynchronization

## Overview
In Actor-Critic models, desynchronization occurs when the Critic updates too fast or slow relative to the Actor, leading to divergence.

## Polyak Averaging Mitigation
```mermaid
graph TD
    A[Critic Loss] --> B[Update Online Critic Network]
    B --> C[Soft-Update Target Critic via Polyak Averaging]
    Note over C: theta_target <- tau * theta_online + (1-tau) * theta_target
```

[← Back to README](../README.md)