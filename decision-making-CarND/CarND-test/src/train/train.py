# coding=utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import socket  # socket module
import json
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Reshape
import matplotlib.pyplot as plt
from math import floor, sqrt
import subprocess
import time
import psutil
import pyautogui
from pynput.mouse import Listener
import os
import pickle
from multiprocessing import Pool
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.compat.v1.Session(config=config))
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"


class DQNAgent:

    def __init__(self, state_height, state_width, action_size):
        self.state_height = state_height
        self.state_width = state_width
        self.action_size = action_size
        self.memory1 = deque(maxlen=2000000)
        self.memory2 = deque(maxlen=2000000)
        # self.memory3 = deque(maxlen=20000)
        self.gamma = 0.90    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.3
        self.epsilon_decay = 0.9  # init with pure exploration
        self.learning_rate = 0.00025
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input1 = Input(shape=(1, self.state_height, self.state_width))
        conv1 = Conv2D(64, (4, 2), strides=1, activation='relu', padding='valid', data_format='channels_first',
                        input_shape=(1, self.state_height, self.state_width))(input1)
        conv2 = Conv2D(64, (4, 2), strides=1, activation='relu', padding='valid')(conv1)
        conv3 = Conv2D(3, 1, strides=1, activation='relu', padding='valid')(conv2)
        flatten = Flatten()(conv3)

        # Adding LSTM layer
        reshaped_flatten = Reshape((1, flatten.shape[1]))(flatten)
        lstm_layer = LSTM(128, activation='relu')(reshaped_flatten)

        input2 = Input(shape=(3,))
        concatenated = concatenate([input2, lstm_layer])

        dense1 = Dense(256, activation='relu')(concatenated)
        dense2 = Dense(64, activation='relu')(dense1)
        output = Dense(self.action_size, activation='linear')(dense2)

        model = Model(inputs=[input1, input2], outputs=output)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))


        # input1 = Input(shape=(1, self.state_height, self.state_width))
        # conv1 = Conv2D(64, (4, 2), strides=1, activation='relu', padding='valid', data_format='channels_first',
        #                  input_shape=(1, self.state_height, self.state_width))(input1)
        # conv2 = Conv2D(64, (4, 2), strides=1, activation='relu', padding='valid')(conv1)
        # conv3 = Conv2D(3, 1, strides=1, activation='relu', padding='valid')(conv2)
        # state1 = Flatten()(conv3)
        # input2 = Input(shape=(3,))
        # state2 = concatenate([input2, state1])
        # state2 = Dense(256, activation='relu')(state2)
        # state2 = Dense(64, activation='relu')(state2)
        # out_put = Dense(self.action_size, activation='linear')(state2)
        # model = Model(inputs=[input1, input2], outputs=out_put)
        # model.compile(loss='mse',
        #               optimizer=Adam(lr=self.learning_rate))

        # model = Sequential()
        # model.add(Conv2D(64, (4, 2), strides=1, activation='relu', padding='valid', data_format='channels_first',
        #                  input_shape=(1, self.state_height, self.state_width)))
        # model.add(Conv2D(64, (4, 2), strides=1, activation='relu', padding='valid'))
        # model.add(Conv2D(3, 1, strides=1, activation='relu', padding='valid'))
        # model.add(Flatten())
        # model.add(Dense(256, activation='relu'))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dense(self.action_size, activation='linear'))
        # model.compile(loss='mse',
        #               optimizer=Adam(lr=self.learning_rate))
        return model



    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember1(self, state, pos, action, reward, next_state, next_pos):
        self.memory1.append((state, pos, action, reward, next_state, next_pos))

    def remember2(self, state, pos, action, reward, next_state, next_pos):
        self.memory2.append((state, pos, action, reward, next_state, next_pos))

    def remember3(self, state, pos, action, reward, next_state, next_pos):
        self.memory3.append((state, pos, action, reward, next_state, next_pos))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            print('taking random', end=" - ")
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        # try:
        minibatch1 = random.sample(self.memory1, int(batch_size / 2))
        minibatch2 = random.sample(self.memory2, batch_size - int(batch_size / 2))
        minibatch = minibatch1 + minibatch2
        
        for state, pos, action, reward, next_state, next_pos in minibatch:
            target = self.model.predict([state, pos])
            t = self.target_model.predict([next_state, next_pos])[0]
            target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit([state, pos], target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
        # except:
        #     with open("debug1.pkl", 'wb') as debug1:
        #         pickle.dump(minibatch1, debug1)
        #     with open("debug2.pkl", 'wb') as debug2:
        #         pickle.dump(minibatch2, debug2)
        #         kill_terminal()
        #         exit(0)


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def connect(ser):
    conn, addr = ser.accept()  # Accepts a TCP connection and returns a new socket and IP address
    print('Connected by', addr)  # Output the client's IP address
    return conn


def parse_log(data):
    # Initialize counters
    left_count = 0
    right_count = 0
    collisions = 0

    # Iterate through each line in the log data
    for line in data.splitlines():
        if 'change_lane_right execute' in line:
            right_count += 1
        elif 'change_lane_left execute' in line:
            left_count += 1
        elif 'collision:' in line:
            collisions += int(line.split(':')[-1])

    return left_count, right_count, collisions

def plot_data(left_count, right_count, collisions, episodeidx):
    # Creating figure and axis objects
    fig, ax = plt.subplots()

    # Data for plotting
    actions = ['Left Lane Change', 'Right Lane Change', 'Collisions']
    counts = [left_count, right_count, collisions]

    # Creating the bar chart
    ax.bar(actions, counts, color=['blue', 'green', 'red'])

    # Adding labels and title
    ax.set_xlabel('Actions')
    ax.set_ylabel('Count')
    ax.set_title('Vehicle Actions Overview')

    # Show the plot
    plt.show(block=False)
    plt.savefig(f'/home/student/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/log/fig/episode{episodeidx}.png')
    plt.pause(3)
    plt.close()

def open_ter(loc, episodeidx):
    try:
        os.system("mkdir -p /home/student/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/log")
    except:
        print("log folder already exists or cannot create log folder")
    try:
        os.system("mkdir -p /home/student/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/log/fig")
    except:
        print("fig folder already exists or cannot create fig folder")
    try:
        os.system(f"touch /home/student/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/log/fig/episode{episodeidx}.png")
    except:
        print("fig file already exists or cannot create fig file")
    try:
        os.system(f"touch /home/student/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/log/episode{episodeidx}.txt")
    except:
        print("txt file already exists or cannot create txt file")
 
    os.system(f"gnome-terminal -e 'bash -c \"cd " + loc + f" && ./path_planning | tee /home/student/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/log/episode{episodeidx}.txt &&exit; exec bash\"'")

    time.sleep(1)
    # return sim


def kill_terminal():
    pids = psutil.pids()
    terminals = []
    for pid in pids:
        p = psutil.Process(pid)
        if p.name() == "gnome-terminal-server":
            terminals.append(pid)
            # os.kill(pid, 9)
    print(f"{terminals=}")
        


def close_all(sim):
    if sim.poll() is None:
        sim.terminate()
        sim.wait()
    time.sleep(2)
    kill_terminal()

def _on_click_(x, y, button, pressed):
    return pressed

EPISODES = 100
location = "/home/student/Autonomous-Driving/decision-making-CarND/CarND-test/build"

HOST = '127.0.0.1'
PORT = 1234
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Define socket type, network communication, TCP
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# server.settimeout(30)
server.bind((HOST, PORT))  # The IP and port of the socket binding
server.listen(1)  # Start TCP listening

state_height = 45
state_width = 3
action_size = 3
agent = DQNAgent(state_height, state_width, action_size)
# agent.epsilon_min = 0.10
# agent.load("./train/episode30.h5")
# with open('./train/exp1.pkl', 'rb') as exp1:
#     agent.memory1 = pickle.load(exp1)
# with open('./train/exp2.pkl', 'rb') as exp2:
#     agent.memory2 = pickle.load(exp2)
batch_size = 16
episode = 1
import os
import time

h5s =      [ file for file in os.listdir("/home/student/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/saved_episode_models") if file.endswith(".h5") ]
h5sctime = [ os.path.getctime(os.path.join("saved_episode_models", file)) for file in os.listdir("/home/student/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/saved_episode_models") if file.endswith(".h5") ]

if len(h5s) > 0:
    max_val = max(h5sctime)
    load_agent = h5s[h5sctime.index(max_val)]
    episode = int(load_agent.split("episode")[1].split(".h5")[0])
    try:
        print(f"/home/student/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/saved_episode_models/episode{episode}.h5")
        agent.load(f"/home/student/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/saved_episode_models/episode{episode}.h5")
        print(f"found agent with episode: {episode} loading that one")
    except:
        print("cannot load the agent starting from scratch")
        agent = DQNAgent(state_height, state_width, action_size)
    try:
        with open(f'/home/student/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/saved_episode_models/exp1forepisode{episode}.pkl', 'rb') as exp1:
            agent.memory1 = pickle.load(exp1)
        print("loaded agent.memory1 from disk")
    except:
        print("cannot load agent memory1")
    try:
        with open(f'/home/student/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/saved_episode_models/exp2forepisode{episode}.pkl', 'rb') as exp2:
            agent.memory2 = pickle.load(exp2)
        print("loaded agent.memory2 from disk")
    except:
        print("cannot load agent memory2")

while episode <= EPISODES:

# for episode in range(0, 101, 10):
#     if episode == 0:
#         episode = 1

    # episode_file = f"episode{episode}.h5"
    # if episode_file in [ file for file in os.listdir("saved_episode_models") if file.endswith('.h5')]:
    #     episode_file = f"/home/student/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/saved_episode_models/{episode_file}"
    #     agent.load(episode_file)
    #     print(f"loading agent episode{episode}.h5")
    # else:
    #     print(f"episode{episode} file not found")

    print(f"{episode=}", end=" - ")
    print(f"{agent.epsilon=}", end=" - ")
    # Start the program
    pool = Pool(processes=2)
    result = []
    result.append(pool.apply_async(connect, (server,)))
    pool.apply_async(open_ter, (location,episode))
    pool.close()
    pool.join()
    conn = result[0].get()
    sim = subprocess.Popen('/home/student/Autonomous-Driving/decision-making-CarND/term3_sim_linux/term3_sim.x86_64')
    while not Listener(on_click=_on_click_):
        pass

    while not Listener(on_click=_on_click_):
        pass
    time.sleep(5)
    pyautogui.click(x=1170, y=876, button='left')
    #okbutton 1170 876
    time.sleep(5)
    pyautogui.click(x=970, y=874, button='left')
    #selectbutton 970 874
    try:
        data = conn.recv(2000)  # Instantiate the received data
    except socket.timeout:
        print("socket timeout the simulator may have crashed")
        close_all(sim)
        continue
    except Exception as e:
        close_all(sim)
        continue
    while not data:
        try:
            data = conn.recv(2000)
        except Exception as e:
            close_all(sim)
            continue
    data = bytes.decode(data)
    # print(data)
    j = json.loads(data)

    # Initialization status information
    # Main car's localization Data
    # car_x = j[1]['x']
    # car_y = j[1]['y']
    car_s = j[1]['s']
    car_d = j[1]['d']
    car_yaw = j[1]['yaw']
    car_speed = j[1]['speed']
    # Sensor Fusion Data, a list of all other cars on the same side of the road.
    sensor_fusion = j[1]['sensor_fusion']
    grid = np.ones((51, 3))
    ego_car_lane = int(floor(car_d/4))
    grid[31:35, ego_car_lane] = car_speed / 100.0

    # sensor_fusion_array = np.array(sensor_fusion)
    for i in range(len(sensor_fusion)):
        vx = sensor_fusion[i][3]
        vy = sensor_fusion[i][4]
        s = sensor_fusion[i][5]
        d = sensor_fusion[i][6]
        check_speed = sqrt(vx * vx + vy * vy)
        car_lane = int(floor(d / 4))
        if 0 <= car_lane < 3:
            s_dis = s - car_s
            if -36 < s_dis < 66:
                pers = - int(floor(s_dis / 2.0)) + 30
                grid[pers:pers + 4, car_lane] = - check_speed / 100.0 * 2.237

    state = np.zeros((state_height, state_width))
    state[:, :] = grid[3:48, :]
    state = np.reshape(state, [-1, 1, state_height, state_width])
    pos = [car_speed / 50, 0, 0]
    if ego_car_lane == 0:
        pos = [car_speed / 50, 0, 1]
    elif ego_car_lane == 1:
        pos = [car_speed / 50, 1, 1]
    elif ego_car_lane == 2:
        pos = [car_speed / 50, 1, 0]
    pos = np.reshape(pos, [1, 3])
    # print(state)
    action = 0
    mess_out = str(action)
    mess_out = str.encode(mess_out)
    conn.sendall(mess_out)
    count = 0
    memcount = 0
    start = time.time()

    # 开始训练过程
    recv_data_count = 0
    while True:
        print(f"{recv_data_count:>3} => ", end=" - ")
        recv_data_count += 1
        # now = time.time()
        # if (now - start) / 60 > 15:
        #     close_all(sim)
        #     break
        try:
            data = conn.recv(2000)
        except Exception as e:
            pass
        while not data:
            try:
                data = conn.recv(2000)
            except Exception as e:
                pass
        # print(f"{data=}", end=" - ")
        data = bytes.decode(data)
        if data == "over":  # End of this iteration
            episode = episode + 1
            print("data over")
            agent.save("/home/student/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/saved_episode_models/episode" + str(episode) + ".h5")
            print("weight saved")
            print("episode: {}, epsilon: {}".format(episode, agent.epsilon))
            with open('/home/student/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/train.txt', 'a') as f:
                f.write(" episode {} epsilon {}\n".format(episode, agent.epsilon))
            close_all(sim)
            conn.close()  # close connection
            with open(f'/home/student/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/saved_episode_models/exp1forepisode{str(episode)}.pkl', 'wb') as exp1:
                pickle.dump(agent.memory1, exp1)
            with open(f'/home/student/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/saved_episode_models/exp2forepisode{str(episode)}.pkl', 'wb') as exp2:
                pickle.dump(agent.memory2, exp2)
            # with open('exp1.pkl', 'rb') as exp1:
            #     agent.memory1 = pickle.load(exp1)
            # with open('exp2.pkl', 'rb') as exp2:
            #     agent.memory2 = pickle.load(exp2)
            # with open('exp3.pkl', 'rb') as exp3:
            #     agent.memory3 = pickle.load(exp3)
            if episode == 41:
                agent.epsilon_min = 0.10
            if episode == 71:
                agent.epsilon_min = 0.03
            if episode == 6:
                agent.epsilon_decay = 0.99985  # start epsilon decay
            print(f"Ending episode {episode-1}")

            break
        try:
            j = json.loads(data)
        except Exception as e:
            close_all(sim)
            break

        # *****************Write program here*****************
        last_state = state
        # print(f"{last_state=}")
        last_pos = pos
        last_act = action
        print(f"{last_act=}", end=" - ")
        last_lane = ego_car_lane
        # **********************************************

        # Main car's localization Data
        # car_x = j[1]['x']
        # car_y = j[1]['y']
        car_s = j[1]['s']
        car_d = j[1]['d']
        car_yaw = j[1]['yaw']
        car_speed = j[1]['speed']
        print(f"{car_s=}", end=" - ")
        if car_speed == 0:
            mess_out = str(0)
            mess_out = str.encode(mess_out)
            conn.sendall(mess_out)
            continue
        # Sensor Fusion Data, a list of all other cars on the same side of the road.
        sensor_fusion = j[1]['sensor_fusion']
        ego_car_lane = int(floor(car_d / 4))
        if last_act == 0:
            last_reward = (2 * ((j[3] - 25.0) / 5.0))  # - abs(ego_car_lane - 1))
        else:
            last_reward = (2 * ((j[3] - 25.0) / 5.0)) - 10.0
        if grid[3:31, last_lane].sum() > 27 and last_act != 0:
            last_reward = -30.0

        grid = np.ones((51, 3))
        grid[31:35, ego_car_lane] = car_speed / 100.0
        # sensor_fusion_array = np.array(sensor_fusion)
        for i in range(len(sensor_fusion)):
            vx = sensor_fusion[i][3]
            vy = sensor_fusion[i][4]
            s = sensor_fusion[i][5]
            d = sensor_fusion[i][6]
            check_speed = sqrt(vx * vx + vy * vy)
            car_lane = int(floor(d / 4))
            if 0 <= car_lane < 3:
                s_dis = s - car_s
                if -36 < s_dis < 66:
                    pers = - int(floor(s_dis / 2.0)) + 30
                    grid[pers:pers + 4, car_lane] = - check_speed / 100.0 * 2.237
            if j[2] < -10:
                last_reward = float(j[2])  # reward -50, -100

        last_reward = last_reward / 10.0
        state = np.zeros((state_height, state_width))
        state[:, :] = grid[3:48, :]
        state = np.reshape(state, [-1, 1, state_height, state_width])
        # print(state)
        pos = [car_speed / 50, 0, 0]
        if ego_car_lane == 0:
            pos = [car_speed / 50, 0, 1]
        elif ego_car_lane == 1:
            pos = [car_speed / 50, 1, 1]
        elif ego_car_lane == 2:
            pos = [car_speed / 50, 1, 0]
        pos = np.reshape(pos, [1, 3])
        print(f"{last_act=}",       end=" - ")
        print(f"{last_reward=:.4}", end=" - ")
        print(f"{car_speed=:.3}",   end=" - ")

        # agent.remember()
        # action = agent.act()
        # *****************Write program here*****************
        if last_act != 0:
            agent.remember1(last_state, last_pos, last_act, last_reward, state, pos)
        else:
            agent.remember2(last_state, last_pos, last_act, last_reward, state, pos)

        action = agent.act([state, pos])
        print(f"{action=}", end=" - ")
        print(f"{count=}", end=" - ")
        print(f"{memcount=} ", end=" - ")
        print(f"{len(agent.memory1)=}", end=" - ")
        # **********************************************

        count += 1
        if count == 10:
            # *****************Write program here*****************
            agent.update_target_model()
            # **********************************************
            print("target model updated", end=" - ")
            count = 0

        if len(agent.memory1) > batch_size and len(agent.memory2) > batch_size and memcount % 100 == 0:
            # *****************Write program here*****************
            print("agent replaying")
            agent.replay(batch_size)
            # agent.memory1.clear()
            # agent.memory2.clear()
            print("agent replayed")
            # **********************************************
        memcount += 1

        mess_out = str(action)
        mess_out = str.encode(mess_out)
        conn.sendall(mess_out)
        print()