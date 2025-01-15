# Robotic Control Project

This repository contains a comprehensive collection of robotics control implementations, including path planning, path tracking, and reinforcement learning-based control algorithms. The project is structured into multiple assignments (Hw1-4) and laboratory exercises (Lab1-2), each focusing on different aspects of robotic control.

## Project Structure

```
.
├── Hw1/         # Basic motion models and path tracking
├── Hw2/         # Advanced control implementations
├── Hw3/         # Extended robotics algorithms
├── Hw4/         # Reinforcement learning for path tracking
├── Lab1/        # Laboratory exercises for motion models
└── Lab2/        # Laboratory exercises for path planning
```

## Features

### Motion Models
- Basic motion model
- Differential drive model
- Bicycle model

### Path Tracking Controllers
- PID Controller
- Pure Pursuit Controller
- Stanley Controller
- LQR Controller

### Path Planning Algorithms
- A* Algorithm
- RRT (Rapidly-exploring Random Tree)
- RRT* (Optimal RRT)

### Reinforcement Learning
- PPO (Proximal Policy Optimization) implementation
- Policy and Value Networks
- Path tracking environment

## Requirements

The project requires the following main dependencies:
- Python 3.x
- PyTorch
- OpenCV (cv2)
- NumPy
- Matplotlib

## Usage

### Basic Navigation and Path Tracking

```bash
python navigation.py [-s SIMULATOR] [-c CONTROLLER] [-p PLANNER] [-m MAP]

Options:
  -s, --simulator    Select simulator type (basic/diff_drive/bicycle)
  -c, --controller   Select controller type (pid/pure_pursuit/stanley/lqr)
  -p, --planner     Select planner type (a_star/rrt/rrt_star)
  -m, --map         Specify map file path
```

### Reinforcement Learning (Hw4)

Training:
```bash
python train.py
```

Evaluation:
```bash
python eval.py
```

Playing with trained model:
```bash
python play.py [--stoch]
```

## Implementation Details

### Simulators
- Basic simulator: Simple point-mass model
- Differential drive: Two-wheeled robot model
- Bicycle model: Car-like robot with steering

### Controllers
- PID: Classical proportional-integral-derivative control
- Pure Pursuit: Geometric path tracking method
- Stanley: Popular steering control method
- LQR: Linear Quadratic Regulator for optimal control

### Path Planners
- A*: Optimal graph search algorithm
- RRT: Sampling-based motion planning
- RRT*: Asymptotically optimal version of RRT

### Reinforcement Learning (Hw4)
- PPO implementation for path tracking
- Custom environment wrapper
- Policy and value network architectures
- Training, evaluation, and visualization tools

## Key Files

- `navigation.py`: Main script for running path planning and tracking
- `train.py`: Training script for reinforcement learning
- `eval.py`: Evaluation script for trained models
- `play.py`: Visualization script for trained models

## Controls

- Mouse Click: Set navigation goal point
- 'R' key: Reset simulation
- 'ESC' key: Exit program

## License

This project is part of an academic course and should be used accordingly.