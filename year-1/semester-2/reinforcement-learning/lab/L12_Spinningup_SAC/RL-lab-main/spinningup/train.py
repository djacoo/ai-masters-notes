import subprocess
import argparse

def train_and_plot():
    ENVIRONMENTS = ["CartPole-v1", "LunarLander-v2", "BipedalWalker-v3", "Pendulum-v0", "Acrobot-v1", "MountainCar-v0", "MountainCarContinuous-v0","FrozenLake-v0"]
    ALGOS = ["ppo", "sac", "ddpg", "trpo", "vpg"]

    parser = argparse.ArgumentParser(description="Train an RL agent using SpinningUp algorithms.")
    parser.add_argument("--env", type=str, choices=ENVIRONMENTS, required=True, help="Environment to train on")
    parser.add_argument("--algo", type=str, choices=ALGOS, required=True, help="RL algorithm to use")
    parser.add_argument("--hid", type=str, default="[128,128]", help="Hidden layer sizes (default: [32,32])")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--epochs", type=str, default="50", help="Number of training epochs (default: 50)")

    args = parser.parse_args()
    
    experiment_data_path = f"spinningup/data/"

    train_cmd = [
        "python", "-m", "spinup.run", args.algo,
        "--hid", args.hid,
        "--env", args.env,
        "--epochs", args.epochs,
        "--exp_name", args.exp_name,
        "--gamma", "0.999",
        "--data_dir", experiment_data_path
    ]
    plot_cmd = [
        "python", "-m", "spinup.run", "plot", f"spinningup/data/{args.exp_name}/{args.exp_name}_s0"
    ]
    
    subprocess.run(train_cmd, check=True)
    subprocess.run(plot_cmd, check=True)

if __name__ == "__main__":
    train_and_plot()
