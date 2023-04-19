import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.utils import common


class DQNAgent(dqn_agent.DqnAgent):
    def __init__(self, observation_spec, action_spec, time_step_spec, learning_rate, gamma):
        q_net = q_network.QNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=(100,)
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        super().__init__(
            time_step_spec,
            action_spec,
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=tf.Variable(0),
            gamma=gamma
        )
