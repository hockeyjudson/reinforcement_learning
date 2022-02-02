#reference : https://keras.io/examples/rl/ddpg_pendulum/
#policy only supports gym pendulum-v1
import tensorflow as tf
import numpy as np
import gym
from collections import deque
import random
from dataclasses import dataclass

@dataclass
class gym_env_data:
    state: object
    action: object
    reward: object
    next_state: object
    done: object

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
    
class DDPG:
    def __init__(self,action_dim,state_dim,action_lower_bound,action_upper_bound,
                 gamma=0.9,tau=0.001,buffer_max_len = 5000,batch_size = 32,policy_num_layers = 2,
                 critic_num_layers = 2,policy_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
                 critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_num_layers = policy_num_layers
        self.critic_num_layers = critic_num_layers
        self.buffer_max_len = buffer_max_len
        self.replay_buffer_queue = deque(maxlen=self.buffer_max_len)
        self.policy_optimizer = policy_optimizer
        self.critic_optimizer = critic_optimizer
        self.lower_bound = action_lower_bound
        self.upper_bound = action_upper_bound
        self.main_policy_network = self.build_policy_network()
        self.target_policy_network = self.build_policy_network()
        self.target_critic_network = self.build_critic_network()
        self.main_critic_network = self.build_critic_network()
        
    def replay_buffer(self,state,action,reward,next_state,done):
        self.replay_buffer_queue.append(gym_env_data(state,action,reward,next_state,done))
    
    def reset_replay_buffer(self):
        self.replay_buffer_queue = deque(maxlen=self.buffer_max_len)
        
    def build_policy_network(self):
        model = tf.keras.Sequential()
        layer = tf.keras.layers.Dense(8,activation='relu')
        model.add((tf.keras.layers.Dense(8,input_shape=(self.state_dim,),name='input')))
        for i in range(self.policy_num_layers):
            model.add(layer)
        model.add(tf.keras.layers.Dense(self.action_dim))
        return model
    
    def build_critic_network(self):
        num_units = 2*(self.state_dim + self.action_dim) + 1
        state_input = tf.keras.Input(shape=self.state_dim)
        state_l1 = tf.keras.layers.Dense(self.state_dim*2,activation="relu")(state_input)
        action_input = tf.keras.Input(shape=self.action_dim)
        action_l1 = tf.keras.layers.Dense(self.action_dim*2,activation="relu")(action_input)
        concat_layer = tf.keras.layers.Concatenate(axis=1)([state_l1,action_l1])
        layer = tf.keras.layers.Dense(num_units,activation="relu")(concat_layer)
        for i in range(self.critic_num_layers):
            layer = tf.keras.layers.Dense(num_units,activation="relu")(layer)
        output_layer = tf.keras.layers.Dense(1)(layer)
        model = tf.keras.models.Model(inputs=[state_input,action_input],outputs=[output_layer])
        return model
    
    def policy(self,state, noise_object):
        sampled_actions = tf.squeeze(self.target_policy_network(state))
        noise = noise_object()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)

        return [np.squeeze(legal_action)]
    
    @tf.function
    def train(self,batch_size):
        minibatch = random.sample(self.replay_buffer_queue,batch_size)
        y_target_list = tf.TensorArray(dtype=tf.float32,size=len(minibatch))
        state_list = tf.TensorArray(dtype=tf.float32,size=len(minibatch))
        action_list = tf.TensorArray(dtype=tf.float32,size=len(minibatch))
        for i,j in enumerate(minibatch):
            y_target_list.write(i,[j.reward + (0.9 * (1 - j.done) * 
                            (self.target_critic_network([j.next_state,
                            self.target_policy_network(j.next_state)])[0]))])
            state_list.write(i,j.state[0])
            action_list.write(i,j.action[0])
            
        #Q-function update one step
        with tf.GradientTape() as tape:
            critic_value = self.main_critic_network([state_list.stack(),action_list.stack()],training=True)
            critic_loss = -tf.math.reduce_mean(tf.math.square(y_target_list.stack() - critic_value))
        critic_grad = tape.gradient(critic_loss, self.main_critic_network.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad,self.main_critic_network.trainable_variables))


        #Policy gradient ascent update one step
        with tf.GradientTape() as tape:
            actions = self.main_policy_network(state_list.stack(),training=True)
            q_value_policy = self.main_critic_network([state_list.stack(),actions],training=True)
            policy_loss = -tf.reduce_mean(q_value_policy)
        policy_grad = tape.gradient(policy_loss,self.main_policy_network.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grad,self.main_policy_network.trainable_variables))
        
    @tf.function
    def update_target(self,target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))