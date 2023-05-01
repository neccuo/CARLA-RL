#!/usr/bin/env python

import pygame
import carla
import random
import numpy as np

pygame.init()

# Set up pygame display
display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)

try:
    # Connect to CARLA server and get the world
    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Attach camera sensor to a vehicle
    vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Spawn camera sensor
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "800")
    camera_bp.set_attribute("image_size_y", "600")
    camera_bp.set_attribute("fov", "90")
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # Listen to camera sensor feed
    camera.listen(lambda image: pygame.surfarray.blit_array(display, np.asarray(image.raw_data).swapaxes(0, 1)))
    # camera.listen(lambda image: pygame.surfarray.blit_array(display, image.raw_data.swapaxes(0, 1)))

    # Start the game loop
    while True:
        pygame.display.flip()
        pygame.event.pump()

finally:
    # camera.destroy()
    # svehicle.destroy()
    pygame.quit()
