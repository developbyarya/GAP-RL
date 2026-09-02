import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--log-std-init", type=float, default=-3.67)
args = parser.parse_args()
print(args.log_std_init)
