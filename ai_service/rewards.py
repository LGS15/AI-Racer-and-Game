#Rewards/Penalties for the racer

PROGRESS_SCALE = 0.02
SPEED_SCALE = 0.0008
STEER_PENALTY = 0.01
COLLISION_PENALTY = 2.0
OFF_TRACK_PENALTY = 0.12
STUCK_PENALTY = 0.02
STUCK_SPEED = 12.0
LAP_BONUS = 8.0

def progress_delta(env):
    delta = env["nextProgressDistance"] - env["previousProgressDistance"]
    if delta < -env["totalLength"]/2:
        delta += env["totalLength"]
    return delta

def compute_rewards(env, action):
    progress = progress_delta(env) * PROGRESS_SCALE
    speed = env["speed"] * SPEED_SCALE
    steer = abs(action["steer"]) * STEER_PENALTY
    collision = COLLISION_PENALTY if env["collision"] else 0.0
    off_track = OFF_TRACK_PENALTY if env["offTrack"] else 0.0
    stuck = STUCK_PENALTY if env["speed"] < STUCK_SPEED else 0.0
    lap = LAP_BONUS if env["lapCompleted"] else 0.0

    return progress + speed + lap - collision - off_track - stuck - steer