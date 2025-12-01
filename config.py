
# Configuration for Bros RL Bot

# Screen Capture Region (top, left, width, height)
# You can use a tool like ShareX or simple screenshot to find these values for your specific game window.
SCREEN_REGION = {"top": 100, "left": 100, "width": 800, "height": 600}

# YOLO Model Path
# Path to the trained YOLO model weights (e.g., 'yolov8n.pt' or custom 'best.pt')
YOLO_MODEL_PATH = "yolov8n.pt"

# Actions
# Map action indices to keys
ACTIONS = {
    0: "left",
    1: "right",
    2: "up",    # Jump / Shoot
    3: "down",  # Duck / Defense
    4: "space", # Action
    5: None     # No-op
}

# Hyperparameters for RL
HYPERPARAMETERS = {
    "total_timesteps": 10000,
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
    "observation_shape": (3, 600, 800), # Channels, Height, Width (Matches SCREEN_REGION)
}
