import argparse
from utils import score_photos

parser = argparse.ArgumentParser(description='Predict or score some picture folder')

parser.add_argument('--mode', type=str, help='set model mode, eval or predict')
parser.add_argument('--folder', type=str, default=None, help='set absolute path to picture folder')
parser.add_argument('--output-folder', type=str, default=None, help='set absolute path to results')
parser.add_argument('--target', type=str, default=None, help='set absolute path to target pickle file')
args = parser.parse_args()

if args.folder is None:
    print('provide -folder argument (example -folder /home/user/photos), see help')
    exit()

if args.mode not in ["predict", "eval", "daemon"]:
    print('provide correct -mode  argument (example -mode eval, or -mode predict, or -mode daemon), see help')
    exit()

score_photos(create_copies=args.mode in ['predict', 'daemon'],
             folder=args.folder,
             output_folder=args.output_folder,
             target=args.target)
