# Advantage Actor-Critic (A2C / A3C)

## Overview
**Actor-Critic** methods reduce variance by introducing an estimator for expected returns (Critic) to baseline the Actor's policy updates.

## Actor-Critic Interaction
```mermaid
graph LR
    Actor[Actor: Policy pi] -- Action --> Env[Environment]
    Env -- State & Reward --> Critic[Critic: Value V]
    Critic -- Advantage A = Q - V --> Actor
```

[← Back to README](../README.md)