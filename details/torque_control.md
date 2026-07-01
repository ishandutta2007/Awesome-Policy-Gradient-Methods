# Continuous Kinetic Torque Control for Humanoid Robotics

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

[← Back to README](../README.md)