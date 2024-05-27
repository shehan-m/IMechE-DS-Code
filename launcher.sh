#!/bin/sh
# launcher.sh
# navigate to home directory, then to this directory, then execute python script, then back home

cd /
cd ~
cd master/pigpio
sudo pigpiod
cd ~
cd IMechE-DS-Code/
sudo python firmware.py
sudo python bbt.py
cd /
