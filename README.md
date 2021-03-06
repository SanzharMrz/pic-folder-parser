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

# Create a venv
$ python3 -m virtualenv venv
$ source venv/bin/activate

# Install requirements
$ pip install -r requirements.txt

# Run bash script for downloading pretrained weights
$ sudo sh get_weights.sh
```
## Usage

Evaluation mode:
```bash
# Run predicting
$ python main.py --mode predict --folder /home/user/photos/ --output-folder /home/user/results/
```

After processing all photos, in output folder will appear __"YES"__ and __"NO"__ subdirectories, where all proceeded pictures were copied. For evaluating models performance, run:

```bash
# Run eval
$ python main.py --mode eval --folder photos/ --target target/photos_target.pickle --output-folder /home/user/results/
```
Check scores.csv in output folder and the same classification report in logs.

Daemon mode:
```bash
# Run daemon
$ python main.py --mode daemon --folder /home/user/photos/ --output-folder /home/user/results/
```

## More params
```bash
--upscale <yes/no> (default: no) - slower but more precise, especially for small faces
--rename <yes/no> (default: yes) - whether or not to rename processed photos
--dynamic-window <int> (deafult: 0) - if > 0 - use dynamic post-processing (to exclude singular errors) (WORKS BADLY ACCORDING TO RAW METRICS)
```

## Usage as a daemon

You can set it up as systemd daemon

```bash
$ nano /etc/systemd/system/photodaemon.service
```

Past this (change working directory to the repo location, python path to your venv location and paths to photo folder and results folder)

```
[Unit]
Description=Photo daemon
After=
Requires=

[Service]
WorkingDirectory=<repo location>
Type=oneshot
RemainAfterExit=yes

ExecStart=<venv location>/bin/python main.py --mode daemon --folder <folder with photos>  --output-folder <output folder>

ExecReload=<venv location>/bin/python main.py --mode daemon --folder <folder with photos>  --output-folder <output folder>

[Install]
WantedBy=multi-user.target
```

Then enable and start the service:
```bash
$ systemctl enable photodaemon
$ systemctl start photodaemon
```
