import os
import time
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame as pg
import numpy as np

from aircraft import Aircraft2D, AircraftConfig
from environment import Environment
from terrain import Terrain
from genetic import GeneticAlgorithm
from controller import Controller


OUT_FOLDER = os.path.join('out', time.strftime('%Y%m%d-%H%M%S'))
os.makedirs(OUT_FOLDER, exist_ok=True)

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

    # Create GA
    ga = GeneticAlgorithm(population_size=200, elite_fraction=0.1, mutation_rate=0.09)
    controllers = [Controller() for _ in range(ga.population_size)]
    episode_time = 30.0  # [s]

    def reset_aircraft() -> list[Aircraft2D]:
        return [Aircraft2D(config, environment, terrain) for _ in range(ga.population_size)]
    
    aircraft = reset_aircraft()
    best_scores = []
    time = 0.0
    sim_speed = 5

    # Main loop
    running = True
    while running:

        screen.fill((135, 206, 235))
        dt = clock.tick(60) / 1000
        time += dt * sim_speed
        fps = clock.get_fps()

        # Control aircraft using GA controllers
        for ac, ctrl in zip(aircraft, controllers):
            for _ in range(sim_speed):
                state = np.array([
                    ac.pos[0],
                    ac.pos[1],
                    ac.vel[0],
                    ac.vel[1],
                    ac.pitch,
                    ac.pitch_rate
                ])
                thrust_cmd, control_surface_cmd, brake_cmd = ctrl.forward(state)
                ac.thrust_setting = thrust_cmd
                ac.control_surface_angle = control_surface_cmd
                ac.wheel_brake = brake_cmd
                ac.step(dt)
            
        # Update camera position (follow best aircraft)
        max_x = max(ac.pos[0] for ac in aircraft if not ac.crashed)
        camera_pos = np.array([min(max_x, terrain.runways[1][1]), camera_pos[1]])

        # Draw terrain
        terrain.draw(screen, camera_pos)

        # Draw aircraft
        for ac in aircraft:
            ac.draw(screen, camera_pos, font)

        # Draw FPS and max X position
        text = font.render(f'FPS: {fps:.0f}', True, (0, 0, 0))
        screen.blit(text, (10, 10))
        text = font.render(f'No. of aircraft: {len(aircraft)}', True, (0, 0, 0))
        screen.blit(text, (10, 30))
        text = font.render(f'Generation: {ga.generation}', True, (0, 0, 0))
        screen.blit(text, (10, 50))
        text = font.render(f'Best X: {max_x:.0f}', True, (0, 0, 0))
        screen.blit(text, (10, 70))
        text = font.render(f'Time: {time:.1f}/{episode_time:.1f} s', True, (0, 0, 0))
        screen.blit(text, (10, 90))
        text = font.render(f'Sim. speed: {sim_speed}x', True, (0, 0, 0))
        screen.blit(text, (10, 110))

        # Update simulation speed with keys
        pressed_keys = pg.key.get_pressed()
        if pressed_keys[pg.K_UP]:
            sim_speed = min(10, sim_speed + 1)
        if pressed_keys[pg.K_DOWN]:
            sim_speed = max(1, sim_speed - 1)

        # Handle events
        pg.display.flip()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            if event.type == pg.VIDEORESIZE:
                screen = pg.display.set_mode((event.w, event.h), pg.RESIZABLE)

        if time >= episode_time or all(ac.crashed for ac in aircraft):
            # Check if aircraft landed correctly
            if any([ac.on_ground and not ac.crashed and ac.vel[0] < 1.0 \
                and ac.pos[0] > terrain.runways[1][0] for ac in aircraft]):
                running = False

            # Calculate scores for each aircraft
            scores = ga.evaluate(aircraft, terrain)
            
            # Save best controller
            filename = f'best_gen{ga.generation}.npz'
            best_controller = controllers[np.argmax(scores)]
            best_controller.save(os.path.join(OUT_FOLDER, filename))
            print(f'Generation {ga.generation} best score: {max(scores):.2f}')
            best_scores.append(max(scores))

            # Create next generation
            controllers = ga.next_generation(controllers, scores)
            aircraft = reset_aircraft()
            time = 0.0
            episode_time += 1.0
            episode_time = min(episode_time, 85.0)

    # Save best scores
    np.savez(os.path.join(OUT_FOLDER, 'generation_scores.npz'), np.array(best_scores))

    pg.quit()


if __name__ == '__main__':
    main()
