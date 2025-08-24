import os
import time
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame as pg
import numpy as np

from aircraft import Aircraft2D, AircraftConfig
from environment import Environment
from terrain import Terrain
from controller import Controller

FILE = 'out\\20250824-184358\\best_gen45.npz'

np.random.seed(1)

def main():
    # Initialize PyGame
    pg.init()
    screen = pg.display.set_mode((1200, 800), pg.RESIZABLE)
    clock = pg.time.Clock()
    camera_pos = np.array([0.0, 150.0])
    font = pg.font.Font(None, 24)
    pg.display.set_caption('Aircraft simulation')

    # Set terrain and environment parameters
    oceans = [(2000, 5000)]
    runways = [(-400, 1400), (5600, 7400)]
    terrain = Terrain(oceans, runways)
    environment = Environment(
        air_density = 1.225,
        gravity = 9.81
    )

    # Set aircraft parameters
    config = AircraftConfig(
        mass = 1000.0,
        max_thrust = 5000.0,
        reference_area = 10.0,
        lift_curve_slope = 5.0,
        parasite_drag_coefficient = 0.02,
        induced_drag_factor = 0.05,
        pitch_rate_gain = 2.0,
        max_control_surface_angle = np.radians(15.0),
        wheel_drag_coefficient = 0.1,
        stall_angle = np.radians(15.0),
        max_vertical_landing_speed = 10.0,
        control_effectiveness_speed = 50.0,
        max_wheel_brake_force = 15000
    )

    # Create aircraft
    aircraft = Aircraft2D(config, environment, terrain)
    controller = Controller.load(FILE)
    time = 0.0

    # Create history
    pos_history = []
    vel_history = []
    pitch_history = []
    thrust_history = []
    control_surface_history = []
    brake_history = []

    # Main loop
    running = True
    while running:

        screen.fill((135, 206, 235))
        dt = clock.tick(60) / 1000
        time += dt
        fps = clock.get_fps()

        # Control aircraft using GA controllers
        state = np.array([
            aircraft.pos[0],
            aircraft.pos[1],
            aircraft.vel[0],
            aircraft.vel[1],
            aircraft.pitch,
            aircraft.pitch_rate
        ])
        thrust_cmd, control_surface_cmd, brake_cmd = controller.forward(state)
        aircraft.thrust_setting = thrust_cmd
        aircraft.control_surface_angle = control_surface_cmd
        aircraft.wheel_brake = brake_cmd
        aircraft.step(dt)
            
        # Update camera position
        camera_pos = np.array([aircraft.pos[0], camera_pos[1]])

        # Draw terrain
        terrain.draw(screen, camera_pos)

        # Draw aircraft
        aircraft.draw(screen, camera_pos, font)

        # Draw FPS and max X position
        text = font.render(f'FPS: {fps:.0f}', True, (0, 0, 0))
        screen.blit(text, (10, 10))
        text = font.render(f'Time: {time:.1f} s', True, (0, 0, 0))
        screen.blit(text, (10, 30))

        # Handle events
        pg.display.flip()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            if event.type == pg.VIDEORESIZE:
                screen = pg.display.set_mode((event.w, event.h), pg.RESIZABLE)

        # Handle end
        if aircraft.on_ground and not aircraft.crashed and aircraft.vel[0] < 1.0 \
            and aircraft.pos[0] > terrain.runways[1][0]:
            running = False

        # Store history
        pos_history.append(aircraft.pos.copy())
        vel_history.append(aircraft.vel.copy())
        pitch_history.append(aircraft.pitch)
        thrust_history.append(aircraft.thrust_setting)
        control_surface_history.append(aircraft.control_surface_angle)
        brake_history.append(aircraft.wheel_brake)

    # Save history
    pos_history = np.array(pos_history)
    vel_history = np.array(vel_history)
    pitch_history = np.array(pitch_history)
    thrust_history = np.array(thrust_history)
    control_surface_history = np.array(control_surface_history)
    brake_history = np.array(brake_history)
    np.savez(os.path.join(os.path.dirname(FILE), 'replay.npz'),
        pos=pos_history,
        vel=vel_history,
        pitch=pitch_history,
        thrust=thrust_history,
        control_surface=control_surface_history,
        brake=brake_history
    )

    pg.quit()


if __name__ == '__main__':
    main()
