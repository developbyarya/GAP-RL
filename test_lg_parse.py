import argparse
import sys
with open("out.txt", "w") as f:
    f.write("Starting...\n")
    from gap_rl.localgrasp.LoG import lg_parse
    f.write("Imported!\n")
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-my-flag", type=str)
    f.write(f"Before lg_parse: {parser._actions}\n")
    parser = lg_parse(parser)
    f.write(f"After lg_parse: {parser._actions}\n")
