#Rewards/Penalties for the racer

PROGRESS_SCALE = 0.03
BACKWARD_PENALTY_SCALE = 0.03
SPEED_SCALE = 0.0002
STEER_PENALTY = 0.01
COLLISION_PENALTY = 2.0
STUCK_PENALTY = 0.02
STUCK_SPEED = 12.0
CHECKPOINT_BONUS = 0.75
LAP_BONUS = 12.0
TIME_PENALTY = 0.003

def progress_delta(env):
    delta = env["nextProgressDistance"] - env["previousProgressDistance"]
    if delta < -env["totalLength"]/2:
        delta += env["totalLength"]
    if delta > env["totalLength"]/2:
        delta -= env["totalLength"]
    return delta

def compute_rewards(env, action):
    delta = progress_delta(env)
    progress = max(0.0, delta) * PROGRESS_SCALE
    backward = max(0.0, -delta) * BACKWARD_PENALTY_SCALE
    speed = env["speed"] * SPEED_SCALE if delta > 0 else 0.0
    steer = abs(action["steer"]) * STEER_PENALTY
    collision = COLLISION_PENALTY if env["collision"] else 0.0
    stuck = STUCK_PENALTY if env["speed"] < STUCK_SPEED else 0.0
    checkpoint = CHECKPOINT_BONUS if env.get("checkpointCompleted") else 0.0
    lap = LAP_BONUS if env["lapCompleted"] else 0.0

    return progress + speed + checkpoint + lap - collision - stuck - steer - backward - TIME_PENALTY
