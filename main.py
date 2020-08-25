import argparse
from utils import score_photos

parser = argparse.ArgumentParser(description='Predict or score some picture folder')

parser.add_argument('--mode', type=str, help='set model mode, eval or predict')
parser.add_argument('--folder', type=str, default=None, help='set absolute path to picture folder')
parser.add_argument('--output-folder', type=str, default=None, help='set absolute path to results')
parser.add_argument('--target', type=str, default=None, help='set absolute path to target pickle file')
parser.add_argument('--dynamic-window', type=int, default=1, help='set +- window to analyze')
parser.add_argument('--rename', type=str, default="yes", help='yes or no')
args = parser.parse_args()

if args.folder is None:
    print('provide -folder argument (example -folder /home/user/photos), see help')
    exit()

if args.mode not in ["predict", "eval", "daemon"]:
    print('provide correct -mode  argument (example -mode eval, or -mode predict, or -mode daemon), see help')
    exit()

while True:
    try:
        score_photos(create_copies=args.mode in ['predict', 'daemon'],
                     folder=args.folder,
                     output_folder=args.output_folder,
                     target=args.target,
                     dynamic_window=args.dynamic_window,
                     do_rename=args.rename == "yes")
    except Exception as e:
        print("Exception:", e)
        if args.mode == "daemon":
            pass
        else:
            exit(1)
    if args.mode != "daemon":
        break
