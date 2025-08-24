"""Controller evaluation functions"""
import numpy as np

from aircraft import Aircraft2D
from terrain import Terrain


CRASH_PENALTY = 30000
STALL_PENALTY = 10000
TAKEOFF_BONUS = 5000
LAND_BONUS = 10000
PHASE_CAP = 20000


def evaluate_aircraft(aircraft: Aircraft2D, terrain: Terrain) -> float:
    """Evaluate the performance of an aircraft with a given controller

    Args:
        aircraft (Aircraft2D): aircraft to evaluate
        terrain (Terrain): terrain for the aircraft to fly over

    Returns:
        float: score
    """
    x, y = aircraft.pos
    vx, vy = aircraft.vel
    approach_dist = 1200

    # Determine flight phase
    if x < terrain.runways[0][1]:
        phase = 0  # takeoff
    elif x < terrain.runways[1][0] - approach_dist:
        phase = 1  # cruise
    elif x < terrain.runways[1][0]:
        phase = 2  # approach
    elif x < terrain.runways[1][1]:
        phase = 3  # landing
    else:
        phase = 4  # overshoot

    # Calculate general penalties
    score = 0.0
    if aircraft.crashed:
        score -= CRASH_PENALTY
        if terrain.runways[1][1] - 10 < x < terrain.runways[1][1] + 10:
            score += CRASH_PENALTY  # negate crash penalty if crash on landing runway
    if aircraft.stalled:
        score -= STALL_PENALTY

    # Phase-specific scoring
    phase_score = 0.0
    if phase == 0:
        # Takeoff
        phase_score += 0 if aircraft.on_ground else TAKEOFF_BONUS
        phase_score -= 0 if aircraft.on_ground else abs(y - 50.0) * 5.0
        phase_score -= max(0.0, 20.0 - vx) * 20.0
        score += np.clip(phase_score, -PHASE_CAP, PHASE_CAP)
    elif phase == 1:
        # Cruise
        score += PHASE_CAP
        phase_score += 2.0 * vx
        phase_score -= abs(y - 100.0) * 5.0
        phase_score -= abs(vy - 90) * 2
        score += np.clip(phase_score, -PHASE_CAP, PHASE_CAP)
    elif phase == 2:
        # Approach
        score += 2 * PHASE_CAP
        target_alt = (terrain.runways[1][0] - x) / approach_dist * 200 \
            if x < terrain.runways[1][0] else 0
        phase_score -= abs(y - target_alt) * 10.0
        score += np.clip(phase_score, -PHASE_CAP, PHASE_CAP)
    elif phase == 3:
        # Landing
        score += 3 * PHASE_CAP
        if not aircraft.on_ground:
            phase_score -= 1000
            phase_score -= y * 100.0
            phase_score -= abs(terrain.runways[1][0] - x)
        else:
            score += LAND_BONUS
            phase_score -= abs(vx) * 20.0
        score += np.clip(phase_score, -PHASE_CAP, PHASE_CAP)
    elif phase == 4:
        # Overshoot
        score -= 20000 + 50 * (x - terrain.runways[1][1])
    else:
        raise ValueError("Invalid flight phase")
    
    return score


    # # Reach target position (and stop there)
    # score -= abs(target_x - x)

    # # Do not crash or stall
    # score -= 20000 if aircraft.crashed else 0
    # score -= 10000 if aircraft.stalled else 0

    # # Start runway approach
    # if x >= 4500 and not aircraft.crashed:
    #     score += 500
    #     score -= abs(y) * 5
    #     score -= abs(vx) * 10

    # # Land smoothly
    # if x >= 5600 and not aircraft.crashed:
    #     score += 500
    #     score -= abs(vy) * 2
    #     score -= abs(aircraft.pitch) * 20
    #     score -= abs(aircraft.pitch_rate) * 10
    #     score += 3000 if aircraft.on_ground else 0

    # # Maintain altitude during cruise
    # if x < 4500:
    #     score -= abs(y - 200) * 2

    # # Takeoff from runway
    # if x < 1400 and not aircraft.crashed and not aircraft.on_ground:
    #     score += 300
    # elif x > 1400:
    #     score += 300

    # # Overshoot penalty
    # if x > 7200:
    #     score += (7200 - x) * 20

    # return score
