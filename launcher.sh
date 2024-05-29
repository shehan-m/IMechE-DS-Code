#!/bin/sh
# launcher.sh
# navigate to home directory, then to this directory, then execute python script, then back home

cd /
cd ~
sudo pigpiod
cd /home/pi/IMechE-DS-Code/
sudo python firmware.py
cd /
