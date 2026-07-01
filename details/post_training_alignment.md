# Post-Training RL Alignment for Reasoning Models

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

[← Back to README](../README.md)