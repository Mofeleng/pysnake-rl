# 🐍 PySnake RL

A Snake game powered by a Deep Q-Learning (DQN) agent built with PyTorch and Pygame. The agent learns to play Snake from scratch through reinforcement learning, improving its strategy over hundreds of games.

---

## How It Works

The agent uses **Deep Q-Learning (DQN)** — a reinforcement learning technique where a neural network learns to predict the best action to take given the current game state.

- The agent observes an 11-value state (danger ahead, food direction, current direction)
- It chooses to go straight, turn right, or turn left
- It receives rewards (+10 for eating food, -10 for dying)
- Over time it learns to associate good actions with higher Q-values

---

## Project Structure

```
pysnake-rl/
├── snake.py       # Pygame Snake game environment
├── agent.py       # DQN agent — state, memory, training loop
├── model.py       # Neural network and Q-trainer
├── plot.py        # Live training progress plot
├── poppins.ttf    # Font used by the game
└── model/
    └── model.pth  # Saved model (created after first best score)
```

---

## Requirements

```
python >= 3.8
torch
pygame
numpy
matplotlib
```

Install dependencies:

```bash
pip install torch pygame numpy matplotlib
```

---

## Running the Project

```bash
python agent.py
```

This will:
- Load a previously saved model if one exists (`model/model.pth`)
- Launch the Snake game window
- Open a live plot tracking score and mean score per game
- Automatically save the model whenever a new high score is achieved

---

## Training Progress

| Phase | Games | What to Expect |
|---|---|---|
| Exploration | 0 – 100 | Mostly random movement, lots of deaths |
| Learning | 100 – 200 | Starts avoiding walls, chasing food |
| Improving | 200+ | Consistent food collection, higher scores |

The mean score line on the plot is the best indicator of genuine improvement — individual game scores will be noisy.

---

## Model

The neural network is a simple two-layer fully connected network:

```
Input (11) → Hidden (256) → Output (3)
```

- **Input**: 11 boolean values representing game state
- **Output**: Q-values for [straight, right, left]
- **Activation**: ReLU on hidden layer
- **Loss**: MSE
- **Optimizer**: Adam (lr=0.01)

### State Representation

| Index | Description |
|---|---|
| 0 | Danger straight |
| 1 | Danger right |
| 2 | Danger left |
| 3–6 | Current direction (L/R/U/D) |
| 7–10 | Food location relative to head |

---

## Key Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `MAX_MEMORY` | 100,000 | Replay buffer size |
| `BATCH_SIZE` | 1,000 | Training batch size |
| `LEARNING_RATE` | 0.01 | Adam optimizer LR |
| `gamma` | 0.9 | Discount factor for future rewards |
| `epsilon` | `max(0, 200 - games)` | Exploration rate |

---

## Model Saving & Loading

- The model is **automatically saved** to `model/model.pth` whenever a new high score is achieved
- On startup, the agent **automatically loads** the saved model if one exists, so training is cumulative across sessions
- To start fresh, delete `model/model.pth`
