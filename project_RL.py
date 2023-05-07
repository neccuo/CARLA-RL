import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from threading import Thread
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriters
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"

MEMORY_FRACTION = 0.4
MIN_REWARD = -200

NUM_EPISODES = 100
MAX_STEPS = 1000

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10
SHOW_PREVIEW = False


# Own Tensorboard class
class ModifiedTensorBoard:
    def __init__(self, log_dir):
        self.writer = SummaryWriters(log_dir=log_dir)
        self.all_write_num = 0

    def log(self, write_name, value, write_num):
        self.writer.add_scalar(write_name, value, write_num)
        self.all_write_num += 1

    def log_histogram(self, write_name, value, write_num, bins=1000):
        self.writer.add_histogram(write_name, value, write_num, bins)
        self.all_write_num += 1

    def log_image(self, write_name, image, write_num):
        self.writer.add_image(write_name, image, write_num)
        self.all_write_num += 1

    def log_graph(self, model, input_to_model=None, verbose=False):
        self.writer.add_graph(model, input_to_model, verbose)
        self.all_write_num += 1

    def close(self):
        self.writer.close()
        
class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        blueprint_library = self.world.get_blueprint_library()
        self.model_3 = blueprint_library.filter("model3")[0]
        self.truck = blueprint_library.filter("carlamotors")[0]
        self.camera_sensor = blueprint_library.find('sensor.camera.rgb')
        self.camera_sensor.set_attribute('image_size_x', f'{self.im_width}')
        self.camera_sensor.set_attribute('image_size_y', f'{self.im_height}')
        self.camera_sensor.set_attribute('fov', '110')

    def reset(self):
        self.collision_hist = []
        self.actor_list = []
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)
        self.rgb_cam = self.world.spawn_actor(
            self.camera_sensor,
            carla.Transform(carla.Location(x=2.5, z=0.7)),
            attach_to=self.vehicle
        )
        self.actor_list.append(self.rgb_cam)
        self.rgb_cam.listen(lambda image: self.process_img(image))
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        self.vehicle.apply_control(carla.VehicleControl(reverse=True))
        time.sleep(3)
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        time.sleep(0.5)
        if self.front_camera is None:
            self.front_camera = self.process_img(np.zeros((self.im_height, self.im_width, 3)))
        return self.front_camera
    
    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1 * self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1 * self.STEER_AMT))
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        return self.front_camera, reward, done, None

    def destroy(self):
        for actor in self.actor_list:
            actor.destroy()
    def get_state(self):
        return self.front_camera
    

class DQNAgent(nn.Module):
    def __init__(self, state_shape, action_size):
        super().__init__()
        self.conv1 = nn.Conv2d(state_shape[2], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.feature_size(state_shape), 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def feature_size(self, state_shape):
        x = torch.zeros(1, *state_shape)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return x.numel()
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float32)
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(3)
        return action
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        
def train():
    agent = DQNAgent(state_shape=(IM_HEIGHT, IM_WIDTH, 3), action_size=3)
    agent.load('best_model.pth')
    env = CarEnv()
    optimizer = optim.Adam(agent.parameters(), lr=0.00001)
    replay_buffer = deque(maxlen=REPLAY_MEMORY_SIZE)
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    for episode in range(NUM_EPISODES):
        total_reward = 0.0
        state = env.reset()
        for step in range(MAX_STEPS):
            action = agent.act([state], epsilon)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            if done:
                break
        if len(replay_buffer) >= TRAINING_BATCH_SIZE:
            batch = random.sample(replay_buffer, TRAINING_BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)
            q_values = agent(states)
            next_q_values = agent(next_states)
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_value = next_q_values.max(1)[0]
            expected_q_value = rewards + DISCOUNT  * next_q_value * (1 - dones)
            loss = (q_value - expected_q_value.detach()).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print(f"Episode {episode} Reward {total_reward} Epsilon {epsilon}")
    agent.save('best_model.pth')
    env.destroy()
    cv2.destroyAllWindows()
    
def test():
    agent = DQNAgent(state_shape=(IM_HEIGHT, IM_WIDTH, 3), action_size=3)
    agent.load('best_model.pth')
    env = CarEnv()
    state = env.reset()
    total_reward = 0.0
    while True:
        action = agent.act([state], epsilon=0.0)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        cv2.imshow("", state)
        cv2.waitKey(1)
        if done:
            break
    print(f"Total reward: {total_reward}")
    env.destroy()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    train()
    #test()
    
    