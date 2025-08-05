from dataclasses import dataclass


@dataclass
class Environment:
    air_density: float  # [kg/m^3]
    gravity: float      # [m/s^2]
