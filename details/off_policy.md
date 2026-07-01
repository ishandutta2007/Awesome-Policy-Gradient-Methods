# Off-Policy Actor-Critic Adaptations

## Overview
Off-policy adaptations use experience replay buffers to recycle transitions gathered by historical policies.

## Replay Architecture
```mermaid
graph LR
    Env[Environment] --> Replay[Replay Buffer]
    Replay -- Sample Batches --> Policy[Target Policy Update]
    Policy --> ImportanceRatio[Importance Sampling Correction]
```

[← Back to README](../README.md)