import numpy as np
from models.dqn_model import DQNAgent
from utils.trading_env import TradingEnv
from utils.trading_utils import plot_learning_curve


def train_dqn(symbol, lot_size, stop_loss, take_profit):
    # Initialize environment and agent
    env = TradingEnv(symbol, lot_size, stop_loss, take_profit)
    state_size = env.get_state().shape[1]
    action_size = 3  # buy, sell, or hold
    agent = DQNAgent(state_size, action_size)

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


train_dqn('EURUSD', 0.01, 50, 100)
