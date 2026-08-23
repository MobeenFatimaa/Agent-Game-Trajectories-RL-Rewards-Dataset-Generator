import json
import random
import numpy as np
import pandas as pd

def generate_exact_rows_rl_trajectories(target_rows=100000):
    np.random.seed(42)
    random.seed(42)
    
    data = []
    agent_types = ["Q-Learning", "PPO", "Random-Walker", "DQN"]
    grid_size = 10
    ep_id = 1
    
    while len(data) < target_rows:
        agent_type = np.random.choice(agent_types, p=[0.3, 0.3, 0.1, 0.3])
        ag_x, ag_y = 0, 0
        target_x, target_y = grid_size - 1, grid_size - 1
        
        policy_quality = {
            "PPO": 0.85,
            "DQN": 0.80,
            "Q-Learning": 0.65,
            "Random-Walker": 0.25
        }[agent_type]
        
        cumulative_reward = 0.0
        step = 1
        
        while True:
            state = {
                "agent_pos": [ag_x, ag_y],
                "target_pos": [target_x, target_y],
                "dist_to_target": abs(ag_x - target_x) + abs(ag_y - target_y)
            }
            
            if np.random.rand() < policy_quality:
                if ag_x < target_x and np.random.rand() > 0.5:
                    action = 1
                elif ag_y < target_y:
                    action = 2
                elif ag_x < target_x:
                    action = 1
                else:
                    action = np.random.choice([0, 3])
            else:
                action = np.random.randint(0, 4)
                
            prev_x, prev_y = ag_x, ag_y
            if action == 0 and ag_y > 0: ag_y -= 1
            elif action == 1 and ag_x < grid_size - 1: ag_x += 1
            elif action == 2 and ag_y < grid_size - 1: ag_y += 1
            elif action == 3 and ag_x > 0: ag_x -= 1
            
            next_state = {
                "agent_pos": [ag_x, ag_y],
                "target_pos": [target_x, target_y],
                "dist_to_target": abs(ag_x - target_x) + abs(ag_y - target_y)
            }
            
            done = False
            if ag_x == target_x and ag_y == target_y:
                step_reward = 100.0
                done = True
            elif ag_x == prev_x and ag_y == prev_y:
                step_reward = -2.0
            else:
                dist_diff = state["dist_to_target"] - next_state["dist_to_target"]
                step_reward = 1.0 if dist_diff > 0 else -1.5
                
            cumulative_reward += step_reward
            entropy = np.round(np.random.beta(2, 5 if agent_type != "Random-Walker" else 1), 4)
            
            data.append({
                "Episode_ID": f"EP-{ep_id:05d}",
                "Step": step,
                "Agent_Type": agent_type,
                "State_JSON": json.dumps(state),
                "Action": action,
                "Action_Name": ["UP", "RIGHT", "DOWN", "LEFT"][action],
                "Reward": round(step_reward, 2),
                "Cumulative_Reward": round(cumulative_reward, 2),
                "Next_State_JSON": json.dumps(next_state),
                "Policy_Entropy": entropy,
                "Done": done
            })
            
            if len(data) == target_rows:
                break
                
            if done or step == 50:
                break
                
            step += 1
            
        ep_id += 1
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    TARGET_ROWS = 100000  # Set your desired exact row count here
    file_path = "synthetic_rl_trajectories.csv"
    
    print(f"Generating dataset with exactly {TARGET_ROWS:,} rows...")
    df = generate_exact_rows_rl_trajectories(target_rows=TARGET_ROWS)
    df.to_csv(file_path, index=False)
    print(f"Done! Saved to '{file_path}'.")