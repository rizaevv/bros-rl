import argparse
import time
from reinforcement_learning.agent import Agent
from reinforcement_learning.environment import BasketballEnv
import config

def main():
    parser = argparse.ArgumentParser(description="Bros RL Basketball Bot")
    parser.add_argument("--mode", type=str, choices=["train", "play"], default="play", help="Mode: train or play")
    parser.add_argument("--timesteps", type=int, default=None, help="Total timesteps for training")
    args = parser.parse_args()

    print("Initializing Basketball RL Bot...")
    
    agent = Agent()

    if args.mode == "train":
        print("Starting training mode...")
        # Give user time to switch to the game window
        print("You have 5 seconds to switch to the game window...")
        time.sleep(5)
        agent.train(total_timesteps=args.timesteps)
        
    elif args.mode == "play":
        print("Starting play mode...")
        # Load model and play
        # Agent already creates an env, so we can use that one or pass one.
        # agent.env is available.
        obs, _ = agent.env.reset()
        
        print("You have 5 seconds to switch to the game window...")
        time.sleep(5)
        
        try:
            while True:
                action = agent.predict(obs)
                obs, reward, done, truncated, info = agent.env.step(action)
                agent.env.render()
                if done or truncated:
                    obs, _ = agent.env.reset()
        except KeyboardInterrupt:
            print("Stopping...")
            agent.env.close()

if __name__ == "__main__":
    main()
