# =============================================================================
# scripts/run_quadruped.py
#
# Open-loop sinusoidal trot gait — no trained policy needed.
# Useful for verifying the XML model is working before training.
#
# Usage:
#   python scripts/run_quadruped.py
#   python scripts/run_quadruped.py --xml robots/quadruped/terrain.xml
# =============================================================================

import argparse
import mujoco
import mujoco.viewer
import numpy as np
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default="robots/quadruped/model.xml")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)

    print(f"Model loaded: {args.xml}")
    print(f"Joints: {model.nq}, Actuators: {model.nu}")
    print("Running open-loop trot gait. Close viewer to stop.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running():
            t     = time.time() - start
            freq  = 2.0
            amp   = 0.4
            phase = np.array([0.0, np.pi, np.pi, 0.0])

            hip_targets  =  amp       * np.sin(2 * np.pi * freq * t + phase)
            knee_targets = -amp * 1.5 * np.abs(np.sin(2 * np.pi * freq * t + phase))

            data.ctrl[0::2] = hip_targets
            data.ctrl[1::2] = knee_targets

            mujoco.mj_step(model, data)

            height = data.qpos[2]
            vx     = data.qvel[0]
            print(f"t={t:.1f}s  height={height:.3f}m  vx={vx:.3f}m/s", end="\r")

            if height < 0.15:
                mujoco.mj_resetData(model, data)
                start = time.time()
                print("\n[reset]")

            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
