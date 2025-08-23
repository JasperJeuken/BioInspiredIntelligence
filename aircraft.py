"""2D aircraft model"""
import numpy as np
import pygame as pg

from dataclasses import dataclass
from typing import Literal

from camera import world_to_screen
from environment import Environment


@dataclass
class AircraftConfig:
    mass: float                         # [kg]
    max_thrust: float                   # [N]
    reference_area: float               # [m^2]
    lift_curve_slope: float             # [1/rad]
    parasite_drag_coefficient: float    # [-]
    induced_drag_factor: float          # [-]
    pitch_rate_gain: float              # [rad/s/rad]
    max_control_surface_angle: float    # [rad]
    wheel_drag_coefficient: float       # [-]
    stall_angle: float                  # [rad]
    max_vertical_landing_speed: float   # [m/s]
    control_effectiveness_speed: float  # [m/s]


FORCES = Literal['lift', 'drag', 'gravity', 'thrust', 'wheel_drag']


class Aircraft2D:
    """2D aircraft model for simulation
    """

    def __init__(self, config: AircraftConfig, environment: Environment) -> None:
        """Initialize the aircraft with specified parameters

        Args:
            config (AircraftConfig): aircraft parameters
            environment (Environment): environment parameters
        """
        self.config: AircraftConfig = config
        self.environment: Environment = environment
        self.color: tuple[int, int, int] = tuple(np.random.randint(0, 256, size=3))

        # State variables
        self._thrust: float = 0.0                    # [-] thrust setting (0.0 to 1.0)
        self._control_surface_angle: float = 0.0     # [rad] flap angle
        self.pos: np.ndarray = np.array([0.0, 0.0])  # [m] position
        self.vel: np.ndarray = np.array([0.0, 0.0])  # [m/s] velocity
        self.pitch: float = 0.0                      # [rad] pitch angle
        self.pitch_rate: float = 0.0                 # [rad/s] pitch rate
        self.forces: dict[FORCES, np.ndarray] = {
            "lift": np.array([0.0, 0.0]),
            "drag": np.array([0.0, 0.0]),
            "gravity": np.array([0.0, 0.0]),
            "thrust": np.array([0.0, 0.0]),
            "wheel_drag": np.array([0.0, 0.0])
        }
        self.stalled: bool = False
        self.on_ground: bool = True
        self.crashed: bool = False

    @property
    def thrust_setting(self) -> float:
        """Get or set the thrust setting of the aircraft in range [0.0, 1.0]

        Returns:
            float: thrust setting [-]
        """
        return self._thrust
    
    @thrust_setting.setter
    def thrust_setting(self, value: float) -> None:
        self._thrust = np.clip(value, 0.0, 1.0)

    @property
    def control_surface_angle(self) -> float:
        """Get or set the control surface angle of the aircraft in range [-max_angle, max_angle]

        Returns:
            float: control surface angle [rad]
        """
        return self._control_surface_angle
    
    @control_surface_angle.setter
    def control_surface_angle(self, value: float) -> None:
        self._control_surface_angle = np.clip(value,
                                              -self.config.max_control_surface_angle,
                                              self.config.max_control_surface_angle)

    @property
    def thrust(self) -> float:
        """Get the current thrust force of the aircraft

        Returns:
            float: current thrust force [N]
        """
        return self._thrust * self.config.max_thrust
    
    @property
    def airspeed(self) -> float:
        """Get the aircraft airspeed

        Returns:
            float: current airspeed [m/s]
        """
        return np.linalg.norm(self.vel)
    
    def calculate_forces(self) -> np.ndarray:
        """Calculate the forces acting on the aircraft

        Returns:
            np.ndarray: total force vector [N]
        """
        # Get velocity unit vector
        v = self.airspeed
        if v > 1e-5:
            vel_unit = self.vel / v
        else:
            vel_unit = np.array([1.0, 0.0])  # default

        # Calculate angle of attack
        flight_path_angle = np.arctan2(vel_unit[1], vel_unit[0])
        alpha = self.pitch - flight_path_angle

        # Calculate lift and drag coefficients
        self.stalled = abs(alpha) > self.config.stall_angle
        if self.stalled:
            lift_coefficient = 0.0
        else:
            lift_coefficient = self.config.lift_curve_slope * alpha
        drag_coefficient = self.config.parasite_drag_coefficient \
            + self.config.induced_drag_factor * lift_coefficient**2
        
        # Calculate lift and drag
        dynamic_pressure = 0.5 * self.environment.air_density * v**2
        lift_mag = dynamic_pressure * self.config.reference_area * lift_coefficient
        drag_mag = dynamic_pressure * self.config.reference_area * drag_coefficient
        lift = lift_mag * np.array([-vel_unit[1], vel_unit[0]])  # assume perpendicular to velocity
        drag = drag_mag * - vel_unit                             # assume opposite to velocity

        # Calculate gravity and thrust
        gravity = np.array([0.0, -self.environment.gravity * self.config.mass])
        thrust = self.thrust * vel_unit  # assume in direction of velocity

        # Calculate wheel drag
        if self.on_ground:
            wheel_drag = -self.config.wheel_drag_coefficient * self.vel
        else:
            wheel_drag = np.array([0.0, 0.0])

        # Store forces
        self.forces["lift"] = lift
        self.forces["drag"] = drag
        self.forces["gravity"] = gravity
        self.forces["thrust"] = thrust
        self.forces["wheel_drag"] = wheel_drag
        return lift + drag + gravity + thrust + wheel_drag
    
    def step(self, dt: float) -> None:
        """Perform a simulation step

        Args:
            dt (float): timestep [s]
        """
        if self.crashed:
            return
        
        # Calculate acceleration
        total_force = self.calculate_forces()
        acceleration = total_force / self.config.mass

        # Update velocity and position
        self.vel += acceleration * dt
        self.pos += self.vel * dt

        # Update pitch angle based on control surface angle
        effectiveness = self.airspeed / (self.airspeed + self.config.control_effectiveness_speed)
        self.pitch_rate = self.config.pitch_rate_gain * self.control_surface_angle * effectiveness
        self.pitch += self.pitch_rate * dt
        self.pitch = np.clip(self.pitch, -np.pi / 2, np.pi / 2)  # limit pitch angle

        # Check crash
        if not self.on_ground and self.pos[1] <= 0.0 and \
            abs(self.vel[1]) > self.config.max_vertical_landing_speed:
            self.crashed = True
            self.vel = np.array([0.0, 0.0])

        # Check ground contact
        if self.pos[1] < 0.0:
            self.pos[1] = 0.0
            if self.vel[1] < 0.0:
                self.vel[1] = 0.0
        self.on_ground = self.pos[1] <= 0.0 and self.vel[1] <= 1e-9

    def draw(self, screen: pg.Surface, camera_pos: np.ndarray) -> None:
        """Draw the aircraft on screen

        Args:
            screen (pg.Surface): PyGame screen
            camera_pos (np.ndarray): camera position in world coordinates
        """
        # Define local aircraft shape
        length = 40
        width = 20
        points = np.array([
            [length / 2, 0],
            [- length / 2, -width / 2],
            [- length / 2, width / 2]
        ])
        
        # Rotate points based on pitch angle
        cos_pitch, sin_pitch = np.cos(self.pitch), np.sin(self.pitch)
        rot_matrix = np.array([[cos_pitch, -sin_pitch], [sin_pitch, cos_pitch]])
        rotated_points = points @ rot_matrix.T
        screen_points = [world_to_screen(self.pos + point, camera_pos, screen.get_size()) 
                         for point in rotated_points]
        
        # Draw aircraft shape
        pg.draw.polygon(screen, self.color, screen_points)
        pg.draw.polygon(screen, (0, 0, 0), screen_points, width=2)
