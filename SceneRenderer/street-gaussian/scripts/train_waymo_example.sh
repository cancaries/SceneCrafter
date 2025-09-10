#!/bin/bash
scenes=("019")
for scene in "${scenes[@]}"; do
    python train.py --config configs/example/waymo_train_$scene.yaml
done
