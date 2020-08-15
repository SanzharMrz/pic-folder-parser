# Folder pictures parsing service

## Description
Use Python? Want to detect some faces? Maybe parse some folders? Looking for elegant solution? This service brings the SOTA face detection architecture to your Python application!

## Requirements

- Python 3.6 or later.
- UNIX OS.
- CUDA GPU (optional).

## Installation
```bash
# Cloning repo
$ git clone https://github.com/SanzharMrz/pic-folder-parser.git

# Step into folder
$ cd pic-folder-parser

# Install requirements
$ pip install -r requirements.txt

# Run bash script for downloading pretrained weights
$ sudo sh get_weights.sh
```
## Usage

Evaluation mode:
```bash
# Run predicting
$ python main.py --mode predict --folder /home/user/photos/  --output-folder /home/user/results/
```

After processing all photos, in output folder __"YES"__ and __"NO"__ subdirectories will appear, where all proceeded pictures were copied. For evaluating models performance, run:

```bash
# Run eval
$ python main.py --mode eval --folder photos/ --target target/photos_target.pickle --output-folder /home/user/results/
```
Check scores.csv in output folder and the same classification report in logs.
