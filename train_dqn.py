import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from utils.trading_env import TradingEnv
from models.dqn_model import DQNAgent


# Set hyperparameters
learning_rate = 1e-3
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.99
min_epsilon = 0.1
target_update_period = 100
batch_size = 64
num_iterations = 2000


# Create the trading environment and wrap in a TF-Agents environment
env = TradingEnv('EURUSD', 0.01, 100, 200)
train_env = tf_py_environment.TFPyEnvironment(env)


# Define the observation and action specs
observation_spec = train_env.observation_spec()
action_spec = train_env.action_spec()


# Create the DQN agent
time_step_spec = train_env.time_step_spec()
agent = DQNAgent(observation_spec, action_spec,
                 time_step_spec, learning_rate, gamma)


# Create the replay buffer
replay_buffer_capacity = 100000
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity
)


# Define the metrics for evaluation
train_metrics = [
    tf_metrics.AverageReturnMetric(batch_size=batch_size),
    tf_metrics.AverageEpisodeLengthMetric(batch_size=batch_size)
]


# Define the training loop
def train():
    global_step = tf.compat.v1.train.get_or_create_global_step()

    # Define the driver
    driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=1
    )

    # Collect initial experience
    initial_collect_policy = agent.collect_policy
    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        initial_collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=100
    )
    initial_collect_driver.run()

    # Reset the agent's policy to the collect policy
    agent.policy = agent.collect_policy

    # Define the dataset
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2
    ).prefetch(3)

    # Define the iterator
    iterator = iter(dataset)

    # Define the metrics
    metrics = tf_metrics.TFMetricsGroup(train_metrics, 'train_metrics')

    # Train the agent
    for iteration in range(num_iterations):
        # Update the epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Evaluate the agent
        if iteration % 100 == 0:
            results = metric_utils.eager_compute(
                metrics,
                train_env,
                agent.policy,
                num_episodes=10,
                train_step=global_step,
            )
            for result in results:
                print(f'{result.name}: {result.result().numpy()}')

        # Collect a trajectory
        driver.run()
        experience, _ = next(iterator)

        # Train the agent
        train_loss = agent.train(experience)
        global_step.assign_add(1)

        # Update the target network
        if global_step.numpy() % target_update_period == 0:
            agent.update_target()

    # Save the final policy
    tf_policy_saver = common.Checkpointer(
        ckpt_dir='./tf_policy',
        max_to_keep=1,
        agent=agent.policy
    )
    tf_policy_saver.save(global_step)
