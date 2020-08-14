import argparse
from utils import score_photos

parser = argparse.ArgumentParser(description='Predict or score some picture folder')

parser.add_argument('-mode', type=str, help='set model mode, eval or predict')
parser.add_argument('-folder', type=str, help='set absolute path to picture folder')
parser.add_argument('-target', type=str, help='set absolute path to target pickle file')
args = parser.parse_args()
    
if args.mode == 'predict':
    if args.folder is None:
        print('provide -folder argument (example -folder /home/user/photos), see help')
        exit()
    score_photos(folder=args.folder, create_copies=True)
elif args.mode == 'eval':    
    if args.target is None:
        print('provide -target argument in eval mode (example -target /home/user/target/20_04.pickle), see help')
        exit()
    score_photos(folder=args.folder, target=args.target,create_copies=False)
else:
    print('first, provide correct -mode  argument (example -mode eval, or -mode predict), see help')
    exit()     
