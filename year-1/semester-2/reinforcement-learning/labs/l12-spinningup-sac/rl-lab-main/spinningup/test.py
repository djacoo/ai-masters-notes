import subprocess
import argparse

def test_agent():
    parser = argparse.ArgumentParser(description="Test a trained RL agent.")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name to test")

    args = parser.parse_args()

    test_cmd = [
        "python", "-m", "spinup.run", "test_policy", f"spinningup/data/{args.exp_name}/{args.exp_name}_s0"
    ]
    subprocess.run(test_cmd, check=True)

if __name__ == "__main__":
    test_agent()
