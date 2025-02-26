import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter
import hockey.hockey_env as h_env
import pickle

import memory
import tools
import temp_agent


"""
This script is used to train a model via self-play.
It can run on the TCML cluster
"""


# Set up environment
env = h_env.HockeyEnv()

ac_space = env.discrete_action_space
o_space = env.observation_space
print(ac_space)
print(o_space)
print(list(zip(env.observation_space.low, env.observation_space.high)))



config = {
    "hidden_sizes": [1024, 1024, 1024],
    "learning_rate": 1e-4,
    "batch_size": 128,
    "iter_fit": 32,
    "max_episodes": 50000,
    "max_steps": 500,
    "buffer_size": int(1e5),
    "discount": 0.95,
    "use_target_net": True,
    "update_target_every": 20,
    "enable_dueling_dqn": True,
    "enable_double_dqn": True,
    "enable_prioritized_replay": True,
    "alpha": 0.6,
    "beta": 0.4,
    "enable_noisy_nets": True,
    "eps": 0.7,
    "eps_decay": 0.9995,
    "eps_min": 0.05,
    "weak_percent": 0.2,
    "self_percent": 0.2,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "eval_every": 200,
    "eval_episodes": 100,
    "print_every": 20
}
load_model = "models/checkpoints/Hockey_DuelingDQN_finetune_run3_cp12000_eps.pt"

run_name = f"Hockey_DuelingDQN_selfplay_run{1}"

agent = temp_agent.DQNAgent(o_space, ac_space, config)
agent.Q.load_state_dict(torch.load(load_model))
agent.update_target()
opponent = h_env.BasicOpponent(weak=False)
writer = SummaryWriter(log_dir=f"logs/{run_name}")

checkpoint_path = "models/checkpoints/" + run_name + "_cp"
checkpoint_idx = 0
checkpoint_count = 2

stats = []
losses = []

last_eval = float("-inf")
side = 0

for i in range(config["max_episodes"]):
    # First explore the environment for one whole episode
    total_reward = 0
    state, _info = env.reset()
    state_agent2 = env.obs_agent_two()
    for t in range(config["max_steps"]):
        done = False
        # Agent chooses action with epsilon-greedy policy
        a_discrete = agent.act(state)
        a = env.discrete_to_continous_action(a_discrete)
        a_agent2_discrete = agent.act(state_agent2)
        a_agent2 = env.discrete_to_continous_action(a_agent2_discrete)
        (state_new, reward, done, trunc, _info) = env.step(np.hstack([a, a_agent2]))
        state_agent2_new = env.obs_agent_two()
        reward_agent_2 = env.get_reward_agent_two(env.get_info_agent_two())

        total_reward += reward
        agent.store_transition((state, a_discrete, reward, state_new, done))
        agent.store_transition((state_agent2, a_agent2_discrete, reward_agent_2, state_agent2_new, done))
        state = state_new
        state_agent2 = state_agent2_new
        if done:
            break

    # Train agent for (iter_fit) iterations
    episode_losses = agent.train()
    losses.extend(episode_losses)
    stats.append(total_reward)

    # Write to tensorboard
    writer.add_scalar("training/loss", np.mean(episode_losses), i)
    writer.add_scalar("training/reward", total_reward, i)
    writer.add_scalar("training/epsilon", agent.eps, i)
    writer.add_scalar("training/steps", t+1, i)

    # Print if necessray
    if i % config["print_every"] == 0:
        print("{}: Done after {} steps. Reward: {}".format(i, t+1, total_reward))

    # Evaluate agent
    if i % config["eval_every"] == 0:
        start_ts = time.time()
        total_reward = 0
        for _ in range(config["eval_episodes"]):
            state, _info = env.reset()
            for t in range(config["max_steps"]):
                a = agent.act(state, eps=0)
                a = env.discrete_to_continous_action(a)
                a_agent2 = opponent.act(env.obs_agent_two())
                (state, reward, done, trunc, _info) = env.step(np.hstack([a, a_agent2]))
                total_reward += reward
                if done:
                    break
        total_reward /= config["eval_episodes"]
        writer.add_scalar("training/eval", total_reward, i)
        print("Evaluation after {} episodes: {} took {}s".format(i, total_reward, (time.time()-start_ts)))

        if total_reward > last_eval:
            percent = (total_reward - last_eval) / last_eval * 100
            print(f"New best model with {percent:.2f}% improvement... Saving model as checkpoint")
            _cp_path = checkpoint_path + str(checkpoint_idx) + ".pt"
            torch.save(agent.Q.state_dict(), _cp_path)
            print(f"Saved model to {_cp_path}")
            checkpoint_idx = (checkpoint_idx + 1) % checkpoint_count
            last_eval = total_reward

    # Save model every 1000 episodes
    if i % 1000 == 0:
        torch.save(agent.Q.state_dict(), checkpoint_path + f"{i}_eps" + ".pt")

# Save model
torch.save(agent.Q.state_dict(), "models/" + run_name + "_final.pt")