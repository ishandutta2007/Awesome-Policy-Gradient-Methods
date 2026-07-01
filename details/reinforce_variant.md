# REINFORCE (Vanilla Policy Gradient)

## Overview
**REINFORCE** scales policy gradient directions by the raw cumulative return ($G_t$). 

## Optimization Formula
$$g = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) G_t \right]$$

## Process Flow
```mermaid
graph TD
    A[Generate Episode] --> B[Calculate Returns G_t]
    B --> C[Compute Log Probabilities]
    C --> D[Apply Gradient Step scaled by G_t]
```

[← Back to README](../README.md)