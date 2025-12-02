
# Configuration for Bros RL Bot

# Screen Capture Region (top, left, width, height)
# You can use a tool like ShareX or simple screenshot to find these values for your specific game window.
SCREEN_REGION = {"top": 100, "left": 100, "width": 800, "height": 600}

# YOLO Model Path
# Path to the trained YOLO model weights
YOLO_MODEL_PATH = "weights/best.pt"

# YOLO Classes (Update these based on your actual training data)
# Example: {0: 'ball', 1: 'hoop', 2: 'player', 3: 'rim'}
YOLO_CLASSES = {
    0: 'ball',
    1: 'hoop',
    2: 'player'
}

# Actions
# Map action indices to keys (Simplified - removed S and G to prevent spam)
ACTIONS = {
    0: "left",   # Move left
    1: "right",  # Move right
    2: "up",     # Jump / Shoot
    3: None      # No-op (do nothing)
}

# Rewards
REWARDS = {
    "dunk": 5.0,
    "3pt": 7.0,
    "steal": 2.0,
    "block": 2.5,
    "opponent_score": -5.0,
    "idle_penalty": -0.2,
    "step_penalty": -0.005,  # Very small penalty to encourage faster play
    "move_to_hoop": 0.05,    # Reward for moving toward target hoop
    "move_away_hoop": -0.02, # Penalty for moving away from hoop
    "ball_possession": 0.05,  # Reward for having ball
    "move_ball_to_hoop": 0.1, # Reward for advancing ball toward hoop
    "move_to_ball": 0.03,     # Reward for moving toward ball
    "dunk_range": 0.1,        # Bonus for being in dunk range
    "defensive_position": 0.02 # Reward for good defensive positioning
}

# Hyperparameters for RL
HYPERPARAMETERS = {
    "total_timesteps": 100000,
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
}

# Environment Settings
ENV_CONFIG = {
    "render_mode": "human", # or "rgb_array"
    "observation_shape": (150, 200, 3), # Height, Width, Channels
    "idle_threshold": 30, # Frames to wait before penalizing idle
}
