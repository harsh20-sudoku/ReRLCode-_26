import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# 1. HYPERPARAMETERS
EPISODES        = 400       # Total training episodes
GAMMA           = 0.99      # Discount factor
LR              = 1e-3      # Learning rate
BATCH_SIZE      = 64        # Mini-batch size for replay
MEMORY_SIZE     = 10_000    # Replay buffer capacity
EPS_START       = 1.0       # Starting exploration rate
EPS_END         = 0.01      # Minimum exploration rate
EPS_DECAY       = 0.995     # Epsilon decay per episode
TARGET_UPDATE   = 10        # Episodes between target network syncs
SOLVE_SCORE     = 195       # CartPole is "solved" at avg 195 over 100 eps

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 2. NEURAL NETWORK (Q-Network)
class QNetwork(nn.Module):
    """
    A simple 3-layer fully-connected network that maps
    states → Q-values for each action.
    """
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# 3. REPLAY BUFFER (Experience Replay)
class ReplayBuffer:
    """
    Stores (state, action, reward, next_state, done) tuples.
    Breaks temporal correlations for stable training.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(DEVICE),
            torch.LongTensor(actions).to(DEVICE),
            torch.FloatTensor(rewards).to(DEVICE),
            torch.FloatTensor(np.array(next_states)).to(DEVICE),
            torch.FloatTensor(dones).to(DEVICE)
        )

    def __len__(self):
        return len(self.buffer)

# 4. DQN AGENT
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.epsilon    = EPS_START

        # Online network (trained every step)
        self.q_net      = QNetwork(state_dim, action_dim).to(DEVICE)
        # Target network (periodically synced — stabilises training)
        self.target_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.q_net.parameters(), lr=LR)
        self.memory     = ReplayBuffer(MEMORY_SIZE)
        self.loss_fn    = nn.MSELoss()

    def select_action(self, state):
        """ε-greedy policy: explore randomly or exploit best known action."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            return self.q_net(state_t).argmax().item()

    def train_step(self):
        """Sample a mini-batch and perform one gradient update."""
        if len(self.memory) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # Current Q-values for taken actions
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values using Bellman equation
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            targets    = rewards + GAMMA * max_next_q * (1 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


# 5. TRAINING LOOP
def train():
    env   = gym.make("CartPole-v1")
    state_dim  = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n               # 2

    agent        = DQNAgent(state_dim, action_dim)
    scores       = []
    avg_scores   = []
    losses       = []

    print(f"\nTraining DQN on CartPole-v1 for {EPISODES} episodes...\n")

    for ep in range(1, EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0
        ep_losses    = []

        while True:
            action              = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done                = terminated or truncated

            agent.memory.push(state, action, reward, next_state, float(done))
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

            state        = next_state
            total_reward += reward

            if done:
                break

        agent.update_epsilon()

        if ep % TARGET_UPDATE == 0:
            agent.sync_target()

        scores.append(total_reward)
        avg100 = np.mean(scores[-100:])
        avg_scores.append(avg100)
        if ep_losses:
            losses.append(np.mean(ep_losses))

        if ep % 50 == 0 or avg100 >= SOLVE_SCORE:
            print(f"Episode {ep:4d} | Score: {total_reward:6.1f} | "
                  f"Avg(100): {avg100:6.2f} | ε: {agent.epsilon:.3f}")

        if avg100 >= SOLVE_SCORE and ep >= 100:
            print(f"\n✅ Solved at episode {ep}! Avg score: {avg100:.2f}")
            break

    env.close()
    return agent, scores, avg_scores, losses

# 6. PLOTTING
def plot_results(scores, avg_scores, losses):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DQN on CartPole-v1", fontsize=14, fontweight='bold')

    # Reward curve
    axes[0].plot(scores,     alpha=0.4, color='steelblue', label='Episode Score')
    axes[0].plot(avg_scores, color='darkorange', linewidth=2, label='Avg (last 100)')
    axes[0].axhline(SOLVE_SCORE, color='green', linestyle='--', label=f'Solve threshold ({SOLVE_SCORE})')
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title("Training Rewards")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss curve
    axes[1].plot(losses, color='crimson', alpha=0.7)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("MSE Loss")
    axes[1].set_title("Training Loss")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("rl_results.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved as rl_results.png")
    plt.show()

# 7. EVALUATION (no exploration)
def evaluate(agent, n_episodes=10):
    env = gym.make("CartPole-v1", render_mode="human")
    print(f"\nEvaluating trained agent for {n_episodes} episodes...")
    eval_scores = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        total    = 0
        while True:
            action = agent.select_action(state)   # epsilon is near 0 now
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        eval_scores.append(total)
        print(f"  Eval episode {ep+1}: {total}")

    env.close()
    print(f"\nMean eval score: {np.mean(eval_scores):.2f}")

# 8. MAIN
if __name__ == "__main__":
    agent, scores, avg_scores, losses = train()
    plot_results(scores, avg_scores, losses)

    # Save the trained model
    torch.save(agent.q_net.state_dict(), "dqn_cartpole.pth")
    print("Model saved as dqn_cartpole.pth")
