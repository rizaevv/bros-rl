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
        self.action_space = spaces.Discrete(len(config.ACTIONS))
        
        # Observation: The screen frame
        self.obs_shape = config.ENV_CONFIG["observation_shape"] # (3, 150, 200)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.obs_shape, dtype=np.uint8)
        
        # State tracking
        self.prev_player_pos = None
        self.prev_ball_pos = None
        self.prev_dist_to_hoop = None
        self.prev_dist_to_ball = None
        self.idle_counter = 0
        self.ball_in_hoop = False
        self.last_shot_distance = 0
        self.steps = 0
        self.ball_possession_frames = 0  # Track how long we have the ball
        self.last_action = None  # Track last action to prevent spam
        self.same_action_counter = 0  # Count consecutive same actions
        self.shot_cooldown = 0  # Prevent immediate shooting

    def step(self, action):
        self.steps += 1
        # Execute action
        action_key = config.ACTIONS.get(action)
        if action_key:
            self.executor.press_key(action_key)
        
        # Capture and process
        frame = self.cap.capture()
        detections = self.detector.get_detections(frame)
        observation = self._process_frame(frame)
        
        # Calculate reward
        reward = self._calculate_reward(detections, action)
        
        # Check if done
        terminated = False # Game over condition
        truncated = False # Time limit
        
        # Placeholder for game over detection (e.g. if we detect "Game Over" text or similar)
        # For now, we rely on manual stop or max steps
        
        info = {"detections": detections}
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset state
        self.prev_player_pos = None
        self.prev_ball_pos = None
        self.prev_dist_to_hoop = None
        self.prev_dist_to_ball = None
        self.idle_counter = 0
        self.ball_in_hoop = False
        self.last_shot_distance = 0
        self.steps = 0
        self.ball_possession_frames = 0
        self.last_action = None
        self.same_action_counter = 0
        self.shot_cooldown = 0
        
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
        # Resize to observation shape (H, W, C)
        # cv2.resize takes (Width, Height)
        resized = cv2.resize(frame, (self.obs_shape[1], self.obs_shape[0]))
        return resized

    def _calculate_reward(self, detections, action):
        reward = config.REWARDS["step_penalty"]
        
        # === ACTION SPAM PREVENTION ===
        # Penalize spamming the same action (especially jump)
        if action == self.last_action and action != 3:  # 3 is no-op, which is okay to repeat
            self.same_action_counter += 1
            # Harsher penalty for jump spam
            if action == 2:  # Jump/shoot action
                if self.same_action_counter > 2:  # Very quick penalty for jump spam
                    reward -= 0.3  # Strong penalty
            else:
                if self.same_action_counter > 5:  # Other actions
                    reward -= 0.1
        else:
            self.same_action_counter = 0
        
        self.last_action = action
        
        # Decrease shot cooldown
        if self.shot_cooldown > 0:
            self.shot_cooldown -= 1
        
        # Get all detections
        players = [d for d in detections if d['class'] == 'player']
        ball = next((d for d in detections if d['class'] == 'ball'), None)
        hoops = [d for d in detections if d['class'] == 'hoop']
        
        # Identify our player (assume closest to bottom of screen or leftmost)
        player = None
        opponent = None
        if len(players) >= 1:
            # Sort by y position (higher y = lower on screen)
            players_sorted = sorted(players, key=lambda p: p['center'][1], reverse=True)
            player = players_sorted[0]  # Our player (bottom)
            if len(players) >= 2:
                opponent = players_sorted[1]  # Opponent
        
        # Identify target hoop (left hoop)
        target_hoop = None
        if hoops:
            # Assume left hoop is the one with smaller x coordinate
            target_hoop = min(hoops, key=lambda h: h['center'][0])
        
        # === 1. MOVEMENT REWARDS ===
        if player and target_hoop:
            p_center = np.array(player['center'])
            h_center = np.array(target_hoop['center'])
            dist_to_hoop = np.linalg.norm(p_center - h_center)
            
            # Reward for getting closer to target hoop
            if self.prev_dist_to_hoop is not None:
                if dist_to_hoop < self.prev_dist_to_hoop:
                    reward += 0.05  # Small reward for moving toward hoop
                elif dist_to_hoop > self.prev_dist_to_hoop:
                    reward -= 0.02  # Small penalty for moving away
            
            self.prev_dist_to_hoop = dist_to_hoop
            
            # Extra reward for being very close to hoop (dunk range)
            if dist_to_hoop < 50:
                reward += 0.1
        
        # === 2. BALL POSSESSION & PROXIMITY ===
        if player and ball:
            p_center = np.array(player['center'])
            b_center = np.array(ball['center'])
            dist_to_ball = np.linalg.norm(p_center - b_center)
            
            # Reward for being close to ball
            if dist_to_ball < 30:  # Close enough to have possession
                self.ball_possession_frames += 1
                reward += 0.05  # Reward for having ball
                
                # Extra reward for moving ball toward hoop while possessing
                if target_hoop and self.prev_ball_pos is not None:
                    h_center = np.array(target_hoop['center'])
                    prev_ball_to_hoop = np.linalg.norm(np.array(self.prev_ball_pos) - h_center)
                    curr_ball_to_hoop = np.linalg.norm(b_center - h_center)
                    
                    if curr_ball_to_hoop < prev_ball_to_hoop:
                        reward += 0.1  # Moving ball toward hoop
            else:
                self.ball_possession_frames = 0
                
                # Reward for moving toward ball when we don't have it
                if self.prev_dist_to_ball is not None:
                    if dist_to_ball < self.prev_dist_to_ball:
                        reward += 0.03
            
            self.prev_dist_to_ball = dist_to_ball
            self.prev_ball_pos = b_center.tolist()
        
        # === 3. IDLE PENALTY ===
        if player:
            current_pos = player['center']
            if self.prev_player_pos:
                dist = np.linalg.norm(np.array(current_pos) - np.array(self.prev_player_pos))
                if dist < 2.0:
                    self.idle_counter += 1
                else:
                    self.idle_counter = 0
            self.prev_player_pos = current_pos
        
        if self.idle_counter > config.ENV_CONFIG["idle_threshold"]:
            reward += config.REWARDS["idle_penalty"]
        
        # === 4. SCORING (Dunk vs 3pt) ===
        # Smart shooting: penalize shooting when far from hoop or on cooldown
        if action == 2:  # UP key = shoot action
            # Default: penalize jumping (must earn it back with good conditions)
            reward -= 0.05  # Small base penalty for any jump
            
            if player and target_hoop:
                p_center = player['center']
                h_center = target_hoop['center']
                dist_to_hoop = np.linalg.norm(np.array(p_center) - np.array(h_center))
                
                # Penalize shooting when too far or not possessing ball
                if dist_to_hoop > 150:  # Too far to shoot
                    reward -= 0.2
                    print("⚠️ Shot too far from hoop!")
                elif self.ball_possession_frames < 3:  # Don't have ball yet
                    reward -= 0.15
                    print("⚠️ Shot without ball!")
                elif self.shot_cooldown > 0:  # On cooldown
                    reward -= 0.1
                    print("⚠️ Shot on cooldown!")
                else:
                    # Good shot attempt - remove base penalty and add small bonus
                    reward += 0.1  # Net reward: +0.05 for good shot timing
                    self.shot_cooldown = 10  # 10 frame cooldown
            else:
                # Can't even detect player/hoop
                reward -= 0.1
        
        # Actual scoring detection
        if ball and target_hoop:
            bx1, by1, bx2, by2 = ball['bbox']
            hx1, hy1, hx2, hy2 = target_hoop['bbox']
            
            # Check overlap
            overlap = not (bx2 < hx1 or bx1 > hx2 or by2 < hy1 or by1 > hy2)
            
            if overlap and not self.ball_in_hoop:
                self.ball_in_hoop = True
                if player:
                    p_center = player['center']
                    h_center = target_hoop['center']
                    dist_to_hoop = np.linalg.norm(np.array(p_center) - np.array(h_center))
                    
                    if dist_to_hoop > 100:
                        reward += config.REWARDS["3pt"]
                        print("🏀 Reward: 3-POINTER!")
                    else:
                        reward += config.REWARDS["dunk"]
                        print("💥 Reward: DUNK!")
                else:
                    reward += config.REWARDS["dunk"]
            elif not overlap:
                self.ball_in_hoop = False
        
        # === 5. STEAL DETECTION ===
        # Only reward steal when near opponent
        if player and opponent and ball:
            p_center = np.array(player['center'])
            o_center = np.array(opponent['center'])
            b_center = np.array(ball['center'])
            
            dist_player_ball = np.linalg.norm(p_center - b_center)
            dist_opponent_ball = np.linalg.norm(o_center - b_center)
            dist_player_opponent = np.linalg.norm(p_center - o_center)
            
            # If we're close to opponent and ball is now closer to us than opponent
            if dist_player_opponent < 50 and dist_player_ball < dist_opponent_ball and dist_player_ball < 30:
                # Likely a steal
                reward += config.REWARDS["steal"]
                print("🔥 Reward: STEAL!")
        
        # === 6. DEFENSIVE REWARDS ===
        # Reward for being between opponent and our hoop (defensive positioning)
        if player and opponent and target_hoop:
            p_center = np.array(player['center'])
            o_center = np.array(opponent['center'])
            h_center = np.array(target_hoop['center'])
            
            # Check if player is between opponent and hoop
            player_to_hoop = np.linalg.norm(p_center - h_center)
            opponent_to_hoop = np.linalg.norm(o_center - h_center)
            
            if player_to_hoop < opponent_to_hoop:
                reward += 0.02  # Small reward for defensive position
        
        # === 7. ACTION DIVERSITY (Maneuvering) ===
        # Penalize spamming same action (encourage varied movement)
        # This would require tracking action history, simplified for now
        
        return reward
