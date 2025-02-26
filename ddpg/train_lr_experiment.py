import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box
import os
import glob
import shutil
import wandb
import pandas as pd
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
import pickle

import hockey.hockey_env as hockey
from memory import Memory
from feedforward import Feedforward
from per_memory import PERMemory

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

class UnsupportedSpace(Exception):
    """Exception for an unsupported observation or action space."""
    def __init__(self, message="Unsupported Space"):
        super().__init__(message)

class QFunction(nn.Module):
    """Q-function that uses a feedforward neural network."""
    def __init__(self, observation_dim, action_dim,
                 hidden_sizes=[100, 100],
                 learning_rate=0.0002):
        super().__init__()
        self.net = Feedforward(
            input_size=observation_dim + action_dim,
            hidden_sizes=hidden_sizes,
            output_size=1
        )
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            eps=1e-6
        )
        self.loss_fn = nn.SmoothL1Loss(reduction='none')
        self.to(device)

    def forward(self, x):
        return self.net(x)

    def fit(self, observations, actions, targets, weights=None):
        self.train()
        self.optimizer.zero_grad()
        pred = self.Q_value(observations, actions)
        loss_unreduced = self.loss_fn(pred, targets)
        if weights is not None:
            loss = (loss_unreduced * weights).mean()
        else:
            loss = loss_unreduced.mean()
        loss.backward()
        self.optimizer.step()
        td_error = pred - targets
        return loss.item(), td_error.detach().cpu().numpy()

    def Q_value(self, observations, actions):
        return self.forward(torch.cat([observations, actions], dim=-1))

class OUNoise:
    """Ornstein-Uhlenbeck noise for exploration."""
    def __init__(self, shape, theta=0.15, dt=1e-2):
        self._shape = shape
        self._theta = theta
        self._dt = dt
        self.noise_prev = np.zeros(self._shape)
        self.reset()

    def __call__(self):
        noise = (
            self.noise_prev
            + self._theta * (-self.noise_prev) * self._dt
            + np.sqrt(self._dt) * np.random.normal(size=self._shape)
        )
        self.noise_prev = noise
        return noise

    def reset(self):
        self.noise_prev = np.zeros(self._shape)

class DDPGAgent:
    """DDPG agent with neural networks for Q and policy. Uses PER if 'use_per' is True."""
    def __init__(self, observation_space, action_space, **userconfig):
        if not isinstance(observation_space, spaces.Box):
            raise UnsupportedSpace(f'Observation space {observation_space} incompatible.')
        if not isinstance(action_space, spaces.Box):
            raise UnsupportedSpace(f'Action space {action_space} incompatible.')

        self.device = device
        self._obs_dim = observation_space.shape[0]
        self._action_dim = 4
        self._action_space = Box(
            low=action_space.low[:4],
            high=action_space.high[:4],
            dtype=np.float32
        )
        self._config = {
            "eps": 0.05,
            "discount": 0.95,
            "buffer_size": int(1e6),
            "batch_size": 512,
            "learning_rate_actor": 0.0003,
            "learning_rate_critic": 0.0003,
            "hidden_sizes_actor": [256, 256],
            "hidden_sizes_critic": [256, 256],
            "update_target_every": 100,
            "use_target_net": True,
            "total_episodes": 50000,
            "seed": 0,
            "tau": 0.005,
            "use_per": True
        }
        self._config.update(userconfig)

        self.eps = self._config["eps"]
        self.discount = self._config["discount"]
        self.batch_size = self._config["batch_size"]
        self.buffer_size = self._config["buffer_size"]
        self.tau = self._config["tau"]
        self.use_target_net = self._config["use_target_net"]
        self.update_target_every = self._config["update_target_every"]
        self.train_iter = 0
        self.use_per = self._config["use_per"]

        if self.use_per:
            self.buffer = PERMemory(
                obs_dim=self._obs_dim,
                act_dim=self._action_dim,
                max_size=self.buffer_size,
                device=self.device
            )
        else:
            self.buffer = Memory(
                obs_dim=self._obs_dim,
                act_dim=self._action_dim,
                max_size=self.buffer_size,
                device=self.device
            )

        self.Q = QFunction(
            observation_dim=self._obs_dim,
            action_dim=self._action_dim,
            hidden_sizes=self._config["hidden_sizes_critic"],
            learning_rate=self._config["learning_rate_critic"]
        )
        self.Q_target = QFunction(
            observation_dim=self._obs_dim,
            action_dim=self._action_dim,
            hidden_sizes=self._config["hidden_sizes_critic"],
            learning_rate=0
        )

        self.policy = Feedforward(
            input_size=self._obs_dim,
            hidden_sizes=self._config["hidden_sizes_actor"],
            output_size=self._action_dim,
            activation_fun=nn.ReLU(),
            output_activation=nn.Tanh()
        )
        self.policy_target = Feedforward(
            input_size=self._obs_dim,
            hidden_sizes=self._config["hidden_sizes_actor"],
            output_size=self._action_dim,
            activation_fun=nn.ReLU(),
            output_activation=nn.Tanh()
        )
        self.policy.to(self.device)
        self.policy_target.to(self.device)

        self._copy_nets()
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=0.0001,
            eps=1e-3
        )
        self.action_noise = OUNoise((self._action_dim,), theta=0.3, dt=0.02)

    def _copy_nets(self):
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())

    def soft_update(self):
        for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
        for target_param, param in zip(self.policy_target.parameters(), self.policy.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def act(self, observation, eps=None):
        if eps is None:
            eps = self.eps
        obs_t = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_t = self.policy(obs_t)
        action_t = action_t.squeeze(0).cpu().numpy()
        noisy_action = action_t + eps * self.action_noise()
        scaled_action = (
            self._action_space.low
            + (noisy_action + 1.0)/2.0
            * (self._action_space.high - self._action_space.low)
        )
        return scaled_action

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def reset(self):
        self.action_noise.reset()

    def train(self, iter_fit=32):
        if self.buffer.get_size() < self.batch_size:
            return []
        losses = []
        self.train_iter += 1
        if self.use_target_net and self.train_iter % self.update_target_every == 0:
            self._copy_nets()

        for _ in range(iter_fit):
            if self.use_per:
                (s, a, rew, s_prime, done, weights, idx) = self.buffer.sample(batch_size=self.batch_size)
            else:
                (s, a, rew, s_prime, done) = self.buffer.sample(batch_size=self.batch_size)
                weights, idx = None, None

            if self.use_target_net:
                q_prime = self.Q_target.Q_value(s_prime, self.policy_target(s_prime))
            else:
                q_prime = self.Q.Q_value(s_prime, self.policy(s_prime))

            td_target = rew + self.discount * (1.0 - done) * q_prime
            loss_val, td_err = self.Q.fit(s, a, td_target, weights=weights)

            self.optimizer.zero_grad()
            q_val = self.Q.Q_value(s, self.policy(s))
            actor_loss = -torch.mean(q_val if weights is None else q_val * weights)
            actor_loss.backward()
            self.optimizer.step()

            if self.use_per and idx is not None:
                td_err = td_err.squeeze(-1)
                self.buffer.update_priorities(idx, td_err)

            losses.append((loss_val, actor_loss.item()))

        self.soft_update()
        return losses

    def save(self, path):
        torch.save({
            "actor": self.policy.state_dict(),
            "critic": self.Q.state_dict(),
            "target_actor": self.policy_target.state_dict(),
            "target_critic": self.Q_target.state_dict(),
            "actor_optimizer": self.optimizer.state_dict()
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(data["actor"])
        self.Q.load_state_dict(data["critic"])
        self.policy_target.load_state_dict(data["target_actor"])
        self.Q_target.load_state_dict(data["target_critic"])
        self.optimizer.load_state_dict(data["actor_optimizer"])

class HockeyEnvRGB(hockey.HockeyEnv):
    """Gymnasium environment that returns an RGB array."""
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50
    }
    def __init__(self, mode="NORMAL"):
        super().__init__(mode=mode)
        self.render_mode = "rgb_array"

    def render(self):
        return super().render(mode="rgb_array")

def store_current_file(destination_folder):
    current_file = __file__
    os.makedirs(destination_folder, exist_ok=True)
    destination_file = os.path.join(destination_folder, os.path.basename(current_file))
    shutil.copy(current_file, destination_file)
    print(f"File saved to {destination_file}")
    
def save_pickle(data, filename):
    pickle_dir = "./pickle_data"
    os.makedirs(pickle_dir, exist_ok=True)
    filepath = os.path.join(pickle_dir, filename)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

def main():
    default_config = {
        "algo": "DDPG",
        "env_name": "HockeyEnv",
        "seed": 0,
        "total_episodes": 20000,
        "learning_rate": 0.0005,
        "batch_size": 1024,
        "gamma": 0.99,
        "tau": 0.005,
        "buffer_size": 300000,
        "exploration_noise": 0.05,
        "touch_bonus": True,
        "use_per": True,
        "opponent_weak": False
    }

    wandb.init(config=default_config, project="Laser_Hockey_Experiments", tags=["LaserHockey"])
    all_params = dict(wandb.config)
    learning_rates = [0.00005, 0.0001,0.0003, 0.0005, 0.001, 0.003]

    all_rewards = {}
    all_win_rates = {}

    for lr in learning_rates:
        all_rewards[str(False)] = []
        all_win_rates[str(True)] = []
    all_rewards[str(False)] = []
    all_win_rates[str(True)] = []
    all_rewards[str(True)] = []
    all_win_rates[str(False)] = []
    for use_per in [False, True]:
        all_params["learning_rate"] = lr
        all_params["learning_rate_actor"] = lr
        all_params["learning_rate_critic"] = lr
        all_params["eps"] = all_params["exploration_noise"]
        all_params["use_per"] = use_per
        run_name = (
            f"algo={all_params['algo']}_"
            f"env={all_params['env_name']}_"
            f"seed={all_params['seed']}_"
            f"eps={all_params['total_episodes']}_"
            f"lr={all_params['learning_rate']}_"
            f"batch={all_params['batch_size']}_"
            f"gamma={all_params['gamma']}_"
            f"tau={all_params['tau']}_"
            f"buf={all_params['buffer_size']}_"
            f"noise={all_params['exploration_noise']}_"
            f"touch={all_params['touch_bonus']}_"
            f"per={all_params['use_per']}_"
            f"oppweak={all_params['opponent_weak']}"
        )



        wandb.run.name = run_name

        torch.manual_seed(all_params["seed"])
        np.random.seed(all_params["seed"])

        video_folder = f"videos_{run_name}"
        checkpoint_folder = f"checkpoints_{run_name}"
        file_used_folder = f"file_used_{run_name}"
        os.makedirs(video_folder, exist_ok=True)
        os.makedirs(checkpoint_folder, exist_ok=True)
        store_current_file(file_used_folder)
        base_env = HockeyEnvRGB(mode="NORMAL")
        base_env.seed(all_params["seed"])

        env = RecordVideo(
            base_env,
            video_folder=video_folder,
            episode_trigger=lambda ep_id: (ep_id % 100) == 0,
            name_prefix="DDPGHockey"
        )





        ddpg = DDPGAgent(env.observation_space, env.action_space, **all_params)
        player2 = hockey.BasicOpponent(weak=all_params["opponent_weak"])
        use_touch_bonus = bool(all_params["touch_bonus"])

        total_rewards = []
        lengths = []
        log_interval = 500
        timestep = 0
        win_count = 0
        wins = []
        win_tracker = []
        running_win_rate = []


        player2 = hockey.BasicOpponent(weak=True)	
        for i_episode in range(1, all_params["total_episodes"] + 1):
            ob, _info = env.reset()
            obs_enemy = env.unwrapped.obs_agent_two()
            ddpg.reset()
            total_reward = 0

            total_t = 500
            for t in range(total_t):
                last_step = False
                if t == total_t-1: 
                    last_step = True

                timestep += 1
                done = False
                a = ddpg.act(ob)
                a2 = player2.act(obs_enemy)
                all_actions = np.hstack([a, a2])
                ob_new, reward, done, trunc, info = env.step(all_actions)

                if use_touch_bonus:
                    reward += info.get('reward_touch_puck', 0)

                total_reward += reward
                ddpg.store_transition((ob, a, reward, ob_new, done))

                ob = ob_new
                obs_enemy = env.unwrapped.obs_agent_two()
                if done or trunc:
                    if "winner" in info and info["winner"] == 1:
                        win_count += 1
                        wins.append(info["winner"])
                    else:
                        wins.append(0)   
                    break
                elif last_step:
                    wins.append(0)
            total_rewards.append(total_reward)
            ddpg.train(iter_fit=8)


            running_window = 1000
            if len(win_tracker) >= running_window:
                avg_win = np.mean(wins[-running_window:]) * 100
                all_win_rates[str(use_per)].append(avg_win)
            else:
                avg_win = np.mean(wins) * 100
                all_win_rates[str(use_per)].append(avg_win)
            running_win_rate.append(avg_win)
            
            all_rewards[str(use_per)].append(total_reward)

            save_pickle(all_rewards, "all_rewards.pkl")
            save_pickle(all_win_rates, "all_win_rates.pkl")

            
            
            
            lengths.append(t)

            winning_percentage = (win_count / i_episode) * 100.0
            

            window_size = 500
            if len(wins) >= window_size:
                recent_win_rate = np.mean(wins[-window_size:]) * 100
            else:
                recent_win_rate = np.mean(wins) * 100

               

            if (i_episode % 500) == 0:
                if recent_win_rate > 50:
                    print("CHANGE TO STRONGER OPPONENT")
                    player2 = hockey.BasicOpponent(weak=False)
                print(f"Recent Win Rate (last {window_size} episodes): {recent_win_rate:.2f}%")
                checkpoint_file = os.path.join(
                    checkpoint_folder,
                    f"checkpoint_ep{i_episode}.pth"
                )
                ddpg.save(checkpoint_file)
                print(f"Checkpoint saved at episode {i_episode}: {checkpoint_file}")
                #wandb.save(checkpoint_file, policy="now")

            if (i_episode % 10000) == 0:
                mp4_files = sorted(
                    glob.glob(os.path.join(video_folder, "*.mp4")),
                    key=os.path.getmtime
                )
                if mp4_files:
                    latest_video = mp4_files[-1]
                    print(f"Episode {i_episode} finished. Uploading video: {latest_video}")
                    wandb.log({
                        "episode_video": wandb.Video(latest_video, caption=f"Episode {i_episode}")
                    })

            if i_episode % log_interval == 0:
                avg_reward = np.mean(total_rewards[-log_interval:])
                avg_length = int(np.mean(lengths[-log_interval:]))
                print(f'Episode {i_episode} avg length: {avg_length} reward: {avg_reward}')
                wandb.log({"avg_reward": avg_reward, "avg_length": avg_length})

    ddpg.save("hockey_agent_final")
    print("Final model saved as hockey_agent_final")
    wandb.finish()
        
            # Ensure we have valid data
    if not all_rewards or not all_win_rates:
        print("Error: No data available for plotting. Check training loop.")

    else:
        plot_dir = "plots"
        os.makedirs(plot_dir, exist_ok=True)
        print(all_win_rates)
        
        # Plot Win Rate over Time for each learning rate
        plt.figure(figsize=(12, 5))
        for lr, win_rates in all_win_rates.items():
            plt.plot(win_rates, label=f"LR {lr}")
        plt.xlabel("Episode")
        plt.ylabel("Win Rate (%)")
        plt.title("Running Win Rate Over Time with and without PER")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(plot_dir, "win_rate_over_time.png"))  # Save plot
        plt.close()

        rolling_window = 100  

        plt.figure(figsize=(12, 5))

        for lr, reward_history in all_rewards.items():
            # Convert to Pandas Series and apply rolling average
            smoothed_rewards = pd.Series(reward_history).rolling(rolling_window).mean()
            
            # Plot only from `rolling_window` onwards to avoid early noise
            plt.plot(range(rolling_window, len(smoothed_rewards) + rolling_window), 
                    smoothed_rewards, label=f"LR {lr}")

        plt.xlabel("Episode")
        plt.ylabel("Smoothed Reward")
        plt.title("Reward per Episode Over Time with and without PER")
        plt.legend()
        plt.savefig(os.path.join(plot_dir, "rewards_over_time.png"))  # Save plot
        plt.close()
        
        


if __name__ == "__main__":
    main()
