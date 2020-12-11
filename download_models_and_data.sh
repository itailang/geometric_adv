#!/bin/bash

# define a download function
function google_drive_download()
{
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

# download models
google_drive_download 1qP9BehCpwrWFX49a5ID5Kqo_P7mUzUoB log.zip
unzip log.zip
rm log.zip

# download data for attack
google_drive_download 1hLoolmRCn4qreH90HM7ujMp_txspFXak eval.zip
mv eval.zip log/autoencoder_victim/
cd log/autoencoder_victim/
unzip eval.zip
rm eval.zip
