# =============================================================================
# scripts/pi_inference.py
#
# Runs the trained policy on a Raspberry Pi.
# No MuJoCo needed — just numpy + stable-baselines3 (or raw numpy weights).
#
# Plug your real robot sensor readings into read_sensors() and
# your motor driver into send_to_motors().
#
# Usage on the Pi:
#   python pi_inference.py --model deploy/final_model.zip --config deploy/policy_info.yaml
# =============================================================================

import argparse
import yaml
import time
import numpy as np

try:
    from stable_baselines3 import PPO, SAC
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("SB3 not installed — using raw numpy inference.")


# ------------------------------------------------------------------------------
# HARDWARE INTERFACE — replace these with your actual sensor/motor code
# ------------------------------------------------------------------------------

def read_sensors() -> np.ndarray:
    """
    Read sensor data from the real robot and return a 33-element observation.

    Replace this with your actual hardware reads:
      - IMU (gyro + accelerometer) -> torso orientation + angular velocity
      - Joint encoders             -> 8 joint angles + 8 joint velocities
      - Force/torque sensors       -> 4 foot contact forces

    The observation must match the training observation exactly:
      qpos (15) + qvel (14) + foot_contacts (4) = 33 values
    """
    # PLACEHOLDER — returns zeros until you wire up real sensors
    obs = np.zeros(33, dtype=np.float32)
    # TODO: fill obs with real sensor readings
    # obs[0:3]  = imu_position()           # torso x, y, z
    # obs[3:7]  = imu_quaternion()          # torso orientation w, x, y, z
    # obs[7:15] = encoder_angles()          # 8 joint angles
    # obs[15:21] = imu_velocity()           # torso linear + angular velocity
    # obs[21:29] = encoder_velocities()     # 8 joint velocities
    # obs[29:33] = foot_contacts() / 100.0  # normalised contact forces
    return obs


def send_to_motors(joint_targets: np.ndarray):
    """
    Send 8 joint angle targets to the robot's servo motors.

    Replace this with your actual motor driver calls:
      - PWM servos: convert radians to PWM pulse width
      - Dynamixel: use the SDK to write goal positions
      - ROS2: publish to /joint_commands topic
    """
    # PLACEHOLDER — prints targets until you wire up real motors
    print(f"Motors: {np.round(joint_targets, 3)}", end="\r")
    # TODO: send joint_targets to your motor controller


# ------------------------------------------------------------------------------
# Main inference loop
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, help="Path to final_model.zip")
    parser.add_argument("--config", required=True, help="Path to policy_info.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        info = yaml.safe_load(f)

    print(f"Robot  : {info['robot']} / {info['task']}")
    print(f"Obs    : {info['obs_size']} values")
    print(f"Actions: {info['action_size']} joints")
    print("Running policy — Ctrl+C to stop.\n")

    # Load policy
    AlgoClass = {"PPO": PPO, "SAC": SAC}[info["algorithm"]]
    model     = AlgoClass.load(args.model)

    ctrl_scale = info["ctrl_scale"]   # 0.5 — same as training
    step_time  = 0.008                # 125 Hz = 8ms per step (4 physics steps * 2ms)

    # Main loop
    step = 0
    while True:
        t_start = time.perf_counter()

        # 1. Read sensors
        obs = read_sensors()

        # 2. Run policy inference
        action, _ = model.predict(obs, deterministic=True)

        # 3. Scale and send to motors
        joint_targets = action * ctrl_scale
        send_to_motors(joint_targets)

        step += 1

        # 4. Sleep to maintain 125 Hz
        elapsed = time.perf_counter() - t_start
        sleep_time = step_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        elif step % 100 == 0:
            print(f"\nWarning: inference took {elapsed*1000:.1f}ms (target: 8ms)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
