from collections import deque
from tensorflow.keras.layers import Input, Conv2D, Flatten, concatenate, Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import os




class DQNAgent:
    def __init__(self, state_height, state_width, action_size):
        self.state_height  = state_height
        self.state_width   = state_width
        self.action_size   = action_size
        self.memory1       = deque(maxlen=2000)
        self.memory2       = deque(maxlen=2000)
        self.gamma         = 0.9
        self.epsilon       = 1.0
        self.epsilon_min   = 0.3
        self.epsilon_decay = 0.9
        self.learning_rate = 0.00025
        self.model         = self.build_model()
        self.target_model  = self.build_model()
        self.update_target_model()
    
    def build_model(self):
        input1 = Input(shape=(1, self.state_height, self.state_width))
        conv1  = Conv2D(64, (4, 2), strides=1, activation='relu', padding='valid', data_format='channels_first', input_shape=(1, self.state_height, self.state_width))(input1)
        conv2  = Conv2D(64, (4, 2), strides=1, activation='relu', padding='valid')(conv1)
        conv3  = Conv2D(3, 1, strides=1, activation='relu', padding='valid')(conv2)
        state1 = Flatten()(conv3)
        input2 = Input(shape=(3,))
        state2 = concatenate([input2, state1])
        state2 = Dense(256, activation='relu')(state2)
        state2 = Dense(64, activation='relu')(state2)
        output = Dense(self.action_size, activation='linear')(state2)
        model  = Model(inputs=[input1, input2], outputs=output)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember1(self, state, pos, action, reward, next_state, next_pos):
        self.memory1.append((state, pos, action, reward, next_state, next_pos))
    
    def remember2(self, state, pos, action, reward, next_state, next_pos):
        self.memory2.append((state, pos, action, reward, next_state, next_pos))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            print("taking random", end=' - ')
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch1 = random.sample(self.memory1, int(batch_size / 2))
        minibatch2 = random.sample(self.memory2, batch_size - int(batch_size / 2))
        minibatch  = minibatch1 + minibatch2

        for state, pos, action, reward, next_state, next_pos in minibatch:
            target1 = self.model.predict([ state, pos ])
            target2 = self.target_model.predict([next_state, next_pos])[0]
            target1[0][action] = reward + self.gamma * np.amax(target2)
            self.model.fit([state, pos], target1, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
    
    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, name):
        self.model.save_weights(name)
    

class AgentTrainer:
    STATE_HEIGHT = 45 # 45 meters ahead
    STATE_WIDTH  = 3  
    ACTION_SIZE  = 3  # number of output lanes

    CARND_DIR       = "/home/student/Autonomous-Driving/decision-making-CarND/CarND-test/"
    CARND_BUILD_DIR = os.path.join(CARND_DIR, "build")
    TRAIN_DIR       = os.path.join(CARND_DIR, "src", "train")
    LOG_DIR         = os.path.join(TRAIN_DIR, "log")
    def __init__(self, ):
        agent = DQNAgent(AgentTrainer.STATE_HEIGHT, AgentTrainer.STATE_WIDTH, AgentTrainer.ACTION_SIZE)


    def connect(self, socket_handle):
        self.connection, self.address = socket_handle.accept()
        print(f"Connected by {self.address}")
    
    def try_making_directory(self, directory_name):
        try:
            os.mkdir("/home/")
        except:
            pass

    def train_episode(episode_number):
        