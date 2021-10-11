import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--A')
parser.add_argument('-v', '--verbose', help='增加输出 verbosity', action='store_true')
args = parser.parse_args()
print(args.A)
if args.verbose:
    print("打开 verbosity")