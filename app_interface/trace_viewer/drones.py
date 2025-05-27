# drones.py
from enum import Enum
from dataclasses import dataclass
from typing import Dict

class DroneType(Enum):
    SELF_MADE = "selfMadeDrone"
    GENERIC_DJI = "DJI"
    MINI_4K = "DJIMini4K"

@dataclass(frozen=True)
class DroneSpec:
    type: DroneType
    sensor_width_mm: float
    focal_length_mm: float

class DroneRegistry:
    _specs: Dict[DroneType, DroneSpec] = {}

    @classmethod
    def register(cls, spec: DroneSpec):
        cls._specs[spec.type] = spec

    @classmethod
    def get(cls, drone_type: DroneType) -> DroneSpec:
        try:
            return cls._specs[drone_type]
        except KeyError:
            raise ValueError(f"No spec for drone type {drone_type}")

# Register built-ins
DroneRegistry.register(DroneSpec(DroneType.SELF_MADE, 5.02, 6))
DroneRegistry.register(DroneSpec(DroneType.GENERIC_DJI, 6.3, 4.5))
DroneRegistry.register(DroneSpec(DroneType.MINI_4K, 6.3 / 4, 4.5))
