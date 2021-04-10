import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from collections import deque
import random

class My_DQN:
    def __init__(self, env, epsilon, gamma, lr, epsilon_decay):
        self.env = env
        # action_space
        self.action_space = env.action_space
        self.num_action_space = self.action_space.n

        # Observation Space
        self.observation_space = env.observation_space
        self.num_observation_space = self.observation_space.shape[0]

        # Variables:
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.counter = 0
        self.rewards_list = []

        # Important things
        self.replay_buffer = deque(maxlen=500000)
        self.batch_size = 64

        self.model = self.initialize_model()
        # self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        # self.loss_fn = tf.keras.losses.mean_squared_error

    def initialize_model(self):
        one = tf.keras.layers.Input(shape=[self.num_observation_space])
        input_layer = tf.keras.layers.Dense(512, activation=tf.keras.activations.relu)(one)
        middle_layer = tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)(input_layer)
        last_layer = tf.keras.layers.Dense(self.num_action_space, activation='linear')(middle_layer)
        model = tf.keras.Model(inputs=[one], outputs=[last_layer])
        model.compile(tf.keras.optimizers.Adam(lr=self.lr),loss=tf.keras.losses.mean_squared_error)
        return model

    def get_action(self, state):
        # Epsilon Greedy policy
        if random.randrange(self.num_action_space) > self.epsilon:
            return np.argmax(self.model.predict(state))

        else:
            return np.random.randint(self.num_action_space)

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update_counter(self):
        self.counter+=1
        step_size = 5
        self.counter = self.counter % step_size

    def get_attributes_from_sample(self, sample):
        states = np.squeeze(np.squeeze(np.array([i[0] for i in sample])))
        actions = np.array([i[1] for i in sample])
        rewards = np.array([i[2] for i in sample])
        next_states = np.squeeze(np.array([i[3] for i in sample]))
        done_list = np.array([i[4] for i in sample])
        return states, actions, rewards, next_states, done_list


    def update_model(self):
        # replay_buffer size Check
        if len(self.replay_buffer) < self.batch_size or self.counter != 0:
            return

        # Early Stopping
        if np.mean(self.rewards_list[-10:]) > 180:
            return

        # take a random sample:
        random_sample = random.sample(self.replay_buffer, self.batch_size)
        # Extract the attributes from sample
        states, actions, rewards, next_states, done_list = self.get_attributes_from_sample(random_sample)

        targets = rewards + self.gamma * (np.max(self.model.predict_on_batch(next_states), axis=1)) * (1 - done_list)
        # print(targets.shape) = (64,)
        # with tf.GradientTape() as tape:
        target_vec = self.model.predict_on_batch(states) # shape = (64, 4)
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec[[indexes], [actions]] = targets

        self.model.fit(states, target_vec, epochs=1, verbose=0)

        # Calculating losses (yet to debug)
    #     losses = tf.reduce_mean(self.loss_fn(np.max(target_vec,axis=1), targets))
    #     print(losses)
    #
    # #Optimize the model:
    # grads = tape.gradient(losses, self.model.trainable_variables)
    # self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


    def learn(self, num_episodes = 2000):
        for episode in range(num_episodes):
            #reset the environment
            state = self.env.reset()

            reward_for_episode = 0
            num_steps = 500
            state = np.reshape(state, [1,self.num_observation_space])
            #what to do in every step
            for step in range(num_steps):
                # Get the action
                received_action = self.get_action(state)

                # Implement the action and the the next_states and rewards
                next_state, reward, done, info = env.step(received_action)

                # Render the actions
                self.env.render()

                # Reshape the next_state and put it in replay buffer
                next_state = np.reshape(next_state, [1, self.num_observation_space])
                # Store the experience in replay_buffer
                self.add_to_replay_buffer(state, received_action, reward, next_state, done)

                # Add rewards
                reward_for_episode+=reward
                # Change the state
                state = next_state

                # Update the model
                self.update_counter()
                self.update_model()

                if done:
                    break

            self.rewards_list.append(reward_for_episode)

            # Decay the epsilon after each completion
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            last_reward_mean = np.mean(self.rewards_list[-100:])
            if last_reward_mean > 200:
                print("DQN Training Complete....")
                break

            # Saving the Model
            # self.model.save('LL1_model.h5', overwrite=True)

            print(f"Episode: {episode} \n Reward: {reward_for_episode} \n Average Reward: {last_reward_mean} \n Epsilon: {self.epsilon}")

    def save(self, name):
        self.model.save(name)

def run_already_trained_model(modelfile_loc, num_episodes=100):
    reward_list = []
    num_test_episode = num_episodes
    env = gym.make('LunarLander-v2')
    print("Starting Testing of the trained model...")

    # Load the model
    trained_model = tf.keras.models.load_model(modelfile_loc)

    step_count = 5000

    for test_episode in range(num_test_episode):
        current_state = env.reset()
        num_observation_space = env.observation_space.shape[0]
        current_state = np.reshape(current_state, [1, num_observation_space])
        # print(f'current_state.shape = {current_state.shape}') = (1,8)
        reward_for_episode = 0
        for step in range(step_count):
            env.render()
            selected_action = np.argmax(trained_model.predict(current_state)[0])
            new_state, reward, done, info = env.step(selected_action)
            new_state = np.reshape(new_state, [1, num_observation_space])
            current_state = new_state
            reward_for_episode+=reward
            if done:
                break
        reward_list.append(reward_for_episode)
        print(f"{test_episode} : Episode || Reward: {reward_for_episode}")

    return reward_list


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    lr = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma = 0.99
    training_episodes = 2000

    '''Use this when training model'''
    # model = My_DQN(env, epsilon,gamma,lr, epsilon_decay)
    # model.learn(training_episodes)

    '''Use this to test the model'''
    reward_list = run_already_trained_model('LL1_model.h5', 2)
    print(reward_list)
    env.close()
