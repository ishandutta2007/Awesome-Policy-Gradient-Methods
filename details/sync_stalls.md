# The Synchronous Mini-Batch Communication Stalls

## Overview
Distributed on-policy training suffers from CPU-to-GPU data transfer bottlenecks during environments rollout gathering.

## GPU Fused Physics Solution
```mermaid
graph LR
    GPUPhys[GPU-based Simulator] -- Direct Memory Access --> GPUBuffers[Tensor Buffers]
    GPUBuffers --> GPULearning[GPU Policy Optimization]
    Note over GPUPhys, GPULearning: Zero CPU-GPU round-trips
```

[← Back to README](../README.md)