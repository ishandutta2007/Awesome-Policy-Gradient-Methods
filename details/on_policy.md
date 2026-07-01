# On-Policy Policy Gradients

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

[← Back to README](../README.md)