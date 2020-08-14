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

```bash
# Run predicting
$ python main.py -m predict -folder /home/user/photos/
```

Note that you can set local path to folder, like:

```bash
# Run predicting
$ python main.py -m predict -folder photos/
```
After processing all photos, in base project directory will appear <b>'results_{folder}'</b>  with __"YES"__ and __"NO"__ subdirectories, where all proceeded pictures were recorded.
