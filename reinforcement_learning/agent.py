from stable_baselines3 import PPO
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from reinforcement_learning.environment import BasketballEnv

class Agent:
    def __init__(self, model_path="models/ppo_basketball"):
        self.model_path = model_path
        self.env = BasketballEnv()
        self.model = None

    def train(self, total_timesteps=None):
        """Trains the agent."""
        timesteps = total_timesteps if total_timesteps else config.HYPERPARAMETERS["total_timesteps"]
        
        print(f"Starting training for {timesteps} timesteps...")
        
        # Check if we can load an existing model to continue training
        if os.path.exists(self.model_path + ".zip"):
            print("Loading existing model...")
            self.model = PPO.load(self.model_path, env=self.env)
        else:
            print("Creating new model...")
            self.model = PPO("CnnPolicy", self.env, verbose=1, 
                             learning_rate=config.HYPERPARAMETERS["learning_rate"],
                             n_steps=config.HYPERPARAMETERS["n_steps"],
                             batch_size=config.HYPERPARAMETERS["batch_size"],
                             n_epochs=config.HYPERPARAMETERS["n_epochs"],
                             gamma=config.HYPERPARAMETERS["gamma"],
                             gae_lambda=config.HYPERPARAMETERS["gae_lambda"],
                             clip_range=config.HYPERPARAMETERS["clip_range"])

        self.model.learn(total_timesteps=timesteps)
        self.save()
        print("Training complete.")

    def predict(self, observation):
        """Predicts the next action."""
        if self.model is None:
             # Try loading
            if os.path.exists(self.model_path + ".zip"):
                self.model = PPO.load(self.model_path)
            else:
                # If no model exists, we can't predict. 
                # For the purpose of the loop, we might return a random action or raise an error.
                # But better to inform the user they need to train first.
                print("Model not found. Returning random action.")
                return self.env.action_space.sample()
        
        action, _states = self.model.predict(observation)
        return action

    def save(self):
        """Saves the model."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
