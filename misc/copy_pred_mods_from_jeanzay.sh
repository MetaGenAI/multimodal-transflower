#!/bin/bash
exp=$1
mkdir inference/generated
mkdir inference/generated/${exp}
mkdir inference/generated/${exp}/predicted_mods
scp -r jeanzay:/gpfswork/rech/imi/usc19dv/mt-lightning/inference/generated/${exp}/predicted_mods/* inference/generated/${exp}/predicted_mods
