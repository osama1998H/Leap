import numpy as np
from trading_env import TradingEnv
from dqn_model import DQNAgent
from training_utils import plot_learning_curve

def train_dqn():
    # Initialize environment and agent
    env = TradingEnv()
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

    # Train agent
    scores = []
    eps_history = []
    n_episodes = 100
    for i in range(n_episodes):
        score = 0
        done = False
        state = env.reset()
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay(32)
            score += reward
            state = next_state
        eps_history.append(agent.epsilon)
        scores.append(score)
        print(f"Episode: {i+1}/{n_episodes}, Score: {score}")

    # Plot learning curve
    plot_learning_curve(scores, eps_history)

    # Save agent model
    agent.save('dqn_model.h5')
