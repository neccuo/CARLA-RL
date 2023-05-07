#!/usr/bin/env python

import glob
import os
import sys
import random
import time
import numpy as np
import math
# import cv2

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

RANDOM_SPAWN = False
SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 800

RWD_SIZE = 10000

SECONDS_PER_EPISODE = 13

class CarEnv:
    STEER_AMT = 0.7
    THROTTLE_AMT = 0.7

    actor_list = []
    collision_hist = []

    # useless
    SHOW_CAM = SHOW_PREVIEW
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.lr = 0.1
        self.gamma = 0.99
        self.epsilon = 0.1

        self.total_reward = None
        self.reward_array = np.ndarray(shape=(RWD_SIZE,), dtype=float)
        self.i = 0


        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        self.actor_list = []
        # 7 states, 3 actions
        self.Q = np.zeros((7, 3))

        blueprint_library = self.world.get_blueprint_library()
        self.model_3 = blueprint_library.filter('model3')[0]

    def reset(self):
        for actor in env.actor_list:
            actor.destroy()

        self.collision_hist = []
        self.actor_list = []

        if self.total_reward != None and self.i < RWD_SIZE:
            self.reward_array[self.i] = np.round(self.total_reward, 2)
            self.i += 1
        self.total_reward = 0

        # FOR RANDOM SPAWN
        if RANDOM_SPAWN:
            self.transform = self.world.get_map().get_spawn_points()
            self.transform = random.choice(self.world.get_map().get_spawn_points())
        else:
            # FIXED SPAWN
            self.transform = self.world.get_map().get_spawn_points()[0]

        self.transform.rotation.yaw += 270

        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        # RGB CAMERA
        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.im_height}')
        self.rgb_cam.set_attribute('fov', '110')

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(2)

        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))

        return self.front_camera
    
    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        #np.save("iout.npy", i)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            print("cv2 is disabled")
            # cv2.imshow("",i3)
            # cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action, yaw):

        # 80 - yaw difference
        reward = 100 - abs(yaw)
        if reward > 0:
            reward *= 5

        if action == 0:
            yaw -= 30
            self.vehicle.apply_control(carla.VehicleControl(throttle=self.THROTTLE_AMT, steer=-1*self.STEER_AMT))
        if action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=self.THROTTLE_AMT, steer=0))
        if action == 2:
            yaw += 30
            self.vehicle.apply_control(carla.VehicleControl(throttle=self.THROTTLE_AMT, steer=1*self.STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward -= 1000
        elif kmh < 20:
            done = False
            reward += -2
        else:
            done = False
            reward += 2
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        # reward for being alive
        if not done:
            reward += 10

        self.total_reward += reward

        return yaw, reward, done, None
    
    def get_state(self, yaw_diff):
        if abs(yaw_diff) < 3:
            return 0
        elif abs(yaw_diff) < 10:
            if yaw_diff > 0:
                return 1
            else:
                return 2
        elif abs(yaw_diff) < 30:
            if yaw_diff > 0:
                return 3
            else:
                return 4
        else:
            if yaw_diff > 0:
                return 5
            else:
                return 6
            
    def get_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 2)
        else:
            return np.argmax(self.Q[state, :])
        
    def learn(self, state, action, reward, next_state):
        self.Q[state, action] = self.Q[state, action] + self.lr * (reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[state, action])
    
    def save(self):
        np.save("qtable.npy", self.Q)
        np.save("reward_array2.npy", self.reward_array)

    def load(self, path="qtable.npy"):
        self.Q = np.load(path)

    def test_rwd_arr(self, path="reward_array.npy"):
        arr = np.load(path)
        for i in range(10):
            print(arr[i])

def transform_to_relevant(transform):
    x = transform.location.x
    y = transform.location.y
    yaw = transform.rotation.yaw
    return x, y, yaw
    
try:

    # vehicle.set_autopilot(True)
    env = CarEnv()

    env.test_rwd_arr()

    env.reset()

    # env.load()

    vehicle = env.vehicle
    map = env.map
    # vehicle.set_autopilot(True)

    vehicle_transform = vehicle.get_transform()
    waypoint = map.get_waypoint(vehicle.get_location())
    target_wp = random.choice(waypoint.next(20.0))

    start_time = time.time()

    while True:
        vehicle_transform = vehicle.get_transform()
        waypoint = map.get_waypoint(vehicle.get_location())

        car_x, car_y, car_yaw = transform_to_relevant(vehicle_transform)
        waypoint_x, waypoint_y, waypoint_yaw = transform_to_relevant(target_wp.transform)

        # print(f"Car transform: {transform_to_relevant(vehicle_transform)}")
        # print(f"Waypoint transform: {transform_to_relevant(target_wp.transform)}")

        yaw_diff = (waypoint_yaw - car_yaw)

        state = env.get_state(yaw_diff)

        action = env.get_action(state)
        new_yaw, reward, done, _ = env.step(action, yaw_diff)

        # if 3 > abs(yaw_diff):
        #     new_yaw, reward, done, _ = env.step(1)
        #     print("go straight")
        # elif yaw_diff > 0:
        #     new_yaw, reward, done, _ = env.step(2)
        #     print("turn right")
        # else:
        #     new_yaw, reward, done, _ = env.step(0)
        #     print("turn left")
        current_time = time.time()
        if current_time - start_time >= 0.5:
            start_time = current_time
            print(f"Reward: {reward}")

        new_state = env.get_state(new_yaw)
        env.learn(state, action, reward, new_state)

        if done:
            print("EPISODE DONE")
            env.reset()
            vehicle = env.vehicle
            vehicle_transform = vehicle.get_transform()
            waypoint = map.get_waypoint(vehicle.get_location())
            target_wp = random.choice(waypoint.next(20.0))


        # print(f"Distance to closest waypoint: {vehicle_wp_dist}")
        # action = random.randint(0, 2)
        # _, reward, done, _ = env.step(action)

        # print(f"Action: {action}")
        # print(f"Reward: {reward}")


        time.sleep(1)
finally:
    for actor in env.actor_list:
        actor.destroy()
    print("All cleaned up!")
    env.save()
    print("Model saved!")

