# =============================================================================
# scripts/start_tensorboard.py
#
# Launches TensorBoard pointing at all experiment logs.
# Bypasses the broken CLI entry point on Windows/Python 3.12.
#
# Usage:
#   python scripts/start_tensorboard.py
# Then open: http://localhost:6006
# =============================================================================

import sys
import glob
import os

# Find all tb_logs folders across all experiments
log_dirs = glob.glob(os.path.join("experiments", "*", "tb_logs"))
if not log_dirs:
    logdir = "./experiments"
else:
    logdir = "./experiments"

print(f"Starting TensorBoard — logdir: {logdir}")
print("Open http://localhost:6006 in your browser.\n")

sys.argv = ["tensorboard", "--logdir", logdir, "--port", "6006"]

from tensorboard import program
tb = program.TensorBoard()
tb.configure(argv=sys.argv)
url = tb.launch()
print(f"TensorBoard running at {url}")
input("Press Enter to stop...\n")
