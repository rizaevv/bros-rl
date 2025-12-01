import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from computer_vision.detector import ScreenCapture, Detector
from automation.key_executor import KeyExecutor

class BasketballEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self):
        super(BasketballEnv, self).__init__()
        
        self.cap = ScreenCapture()
        self.detector = Detector()
        self.executor = KeyExecutor()
        
        # Define action and observation space
        # Actions: left, right, up, down, space, no-op
        self.action_space = spaces.Discrete(len(config.ACTIONS))
        
        # Observation: The screen frame (simplified for now)
        # We'll resize to a smaller resolution for the agent to process faster
        self.obs_shape = (150, 200, 3) # Height, Width, Channels
        self.observation_space = spaces.Box(low=0, high=255, shape=self.obs_shape, dtype=np.uint8)

    def step(self, action):
        # Execute action
        action_key = config.ACTIONS.get(action)
        if action_key:
            self.executor.press_key(action_key)
        
        # Wait a bit for action to take effect
        # time.sleep(0.05) 
        
        # Get observation
        frame = self.cap.capture()
        observation = self._process_frame(frame)
        
        # Calculate reward
        # Placeholder: In a real scenario, we need to detect score changes or game over
        reward = 0 
        
        # Check if done
        # Placeholder: Need a way to detect game over
        done = False
        truncated = False
        info = {}
        
        return observation, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the game state if possible (e.g., press restart button)
        # For now, just capture the screen
        frame = self.cap.capture()
        observation = self._process_frame(frame)
        info = {}
        return observation, info

    def render(self, mode='human'):
        if mode == 'human':
            # We can show the captured frame with cv2
            pass
        elif mode == 'rgb_array':
            return self.cap.capture()

    def close(self):
        cv2.destroyAllWindows()

    def _process_frame(self, frame):
        # Resize to observation shape
        resized = cv2.resize(frame, (self.obs_shape[1], self.obs_shape[0]))
        return resized
