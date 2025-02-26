from __future__ import annotations

import argparse
import uuid

import hockey.hockey_env as h_env
import numpy as np

import torch
import torch.nn.functional as F


from comprl.client import Agent, launch_client


class RandomAgent(Agent):
    """A hockey agent that simply uses random actions."""

    def get_step(self, observation: list[float]) -> list[float]:
        return np.random.uniform(-1, 1, 4).tolist()

    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, enable_dueling_dqn=False, device = 'cpu'):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.enable_dueling_dqn = enable_dueling_dqn
        self.device = device

        if self.enable_dueling_dqn:
            # cut the last hidden layer to the advantage and value streams
            self.dueling_size = self.hidden_sizes[-1]
            self.hidden_sizes = self.hidden_sizes[:-1]

        layers = []
        in_size = self.input_size
        for h in hidden_sizes:
            layers.append(torch.nn.Linear(in_size, h))
            layers.append(torch.nn.ReLU())
            in_size = h
        

        if self.enable_dueling_dqn:
            # Value steram
            self.fc_value = torch.nn.Linear(in_size, self.dueling_size)
            self.value = torch.nn.Linear(self.dueling_size, 1)

            # Advantages stream
            self.fc_advantages = torch.nn.Linear(in_size, self.dueling_size)
            self.advantages = torch.nn.Linear(self.dueling_size, self.output_size)
        else:
            layers.append(torch.nn.Linear(in_size, output_size))

        self.fully_connected = torch.nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, x):
        '''
        Returns [batch_size, action_space_size]
        '''
        x = self.fully_connected(x)
        if self.enable_dueling_dqn:
            # Value calculation
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            # Advantages calculation
            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            # Calculate Q
            Q = V + A - torch.mean(A, dim=-1, keepdim=True)
        
        else:
            Q = x

        return Q
            
    
    def predict(self, x):
        '''
        Runs without gradients and takes and returns numpy arrays
        '''
        x = torch.from_numpy(x).float().to(self.device)
        self.eval()
        with torch.no_grad():
            out = self.forward(x).cpu().numpy()
        self.train()
        return out
    
class QFunction(Feedforward):
    def __init__(self, state_dim, action_dim, hidden_sizes, learning_rate, enable_dueling_dqn=False, device = 'cpu'):
        super().__init__(input_size=state_dim,
                         hidden_sizes=hidden_sizes,
                         output_size=action_dim,
                         enable_dueling_dqn=enable_dueling_dqn,
                         device=device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = torch.nn.SmoothL1Loss()

    def fit(self, states, actions, targets):
        self.train()
        self.optimizer.zero_grad()

        # Forward pass
        acts = torch.from_numpy(actions).to(self.device)
        pred = self.Q_value(torch.from_numpy(states).float().to(self.device), acts)
        loss = self.loss(pred, torch.from_numpy(targets).float().to(self.device))

        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def Q_value(self, states, actions):
        return self.forward(states).gather(1, actions[:, None])
    
    def maxQ(self, states):
        return np.max(self.predict(states), axis=-1, keepdims=True)
    
    def greedy_action(self, states):
        return np.argmax(self.predict(states), axis=-1)

keep_mode = True

def discrete_to_continous_action(discrete_action):
    ''' converts discrete actions into continuous ones (for one player)
        The actions allow only one operation each timestep, e.g. X or Y or angle change.
        This is surely limiting. Other discrete actions are possible
        Action 0: do nothing
        Action 1: -1 in x
        Action 2: 1 in x
        Action 3: -1 in y
        Action 4: 1 in y
        Action 5: -1 in angle
        Action 6: 1 in angle
        Action 7: shoot (if keep_mode is on)
        '''
    action_cont = [(discrete_action == 1) * -1.0 + (discrete_action == 2) * 1.0,  # player x
                   (discrete_action == 3) * -1.0 + (discrete_action == 4) * 1.0,  # player y
                   (discrete_action == 5) * -1.0 + (discrete_action == 6) * 1.0]  # player angle
    if keep_mode:
      action_cont.append((discrete_action == 7) * 1.0)

    return action_cont


class HockeyAgent(Agent):
    """A hockey agent that can be weak or strong."""

    def __init__(self, weak: bool) -> None:
        super().__init__()

        self.hockey_agent = h_env.BasicOpponent(weak=weak)

    def get_step(self, observation: list[float]) -> list[float]:
        # NOTE: If your agent is using discrete actions (0-7), you can use
        # HockeyEnv.discrete_to_continous_action to convert the action:
        #
        # from hockey.hockey_env import HockeyEnv
        # env = HockeyEnv()
        # continuous_action = env.discrete_to_continous_action(discrete_action)

        action = self.hockey_agent.act(observation).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )

class DDQNAgent(Agent):
    def __init__(self,) -> None:
        super().__init__()
        self.Q= QFunction(state_dim=18, action_dim=7, hidden_sizes=[256, 256, 256], learning_rate=0.001, enable_dueling_dqn=True)
        # model_path = "Hockey_DuelingDQN_finetune_run3_cp12000_eps.pt"
        # model_path = "Hockey_DuelingDQN_finetune_run5_cp22000_eps.pt"
        # model_path = "Hockey_DuelingDQN_finetune_run5_cp200000_eps.pt"
        model_path = "Hockey_DuelingDQN_train_both_run2_cp3.pt"
        self.Q.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        #print("Started DDQN agent")

    def act(self, state):
        return discrete_to_continous_action(self.Q.greedy_action(state))

    def get_step(self, observation: list[float]) -> list[float]:
        action = self.act(np.array(observation))
        #print(len(discrete_to_continous_action(action)))
        return discrete_to_continous_action(action)
    
    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )



# Function to initialize the agent.  This function is used with `launch_client` below,
# to lauch the client and connect to the server.
def initialize_agent(agent_args: list[str]) -> Agent:
    # Use argparse to parse the arguments given in `agent_args`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["weak", "strong", "random", "DDQN"],
        default="weak",
        help="Which agent to use.",
    )
    args = parser.parse_args(agent_args)

    # Initialize the agent based on the arguments.
    agent: Agent
    if args.agent == "weak":
        agent = HockeyAgent(weak=True)
    elif args.agent == "strong":
        agent = HockeyAgent(weak=False)
    elif args.agent == "random":
        agent = RandomAgent()
    elif args.agent == "DDQN":
        agent = DDQNAgent()
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # And finally return the agent.
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
