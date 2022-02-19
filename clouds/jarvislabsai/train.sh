# . ../sg/bin/activate

python train.py --data=../processed --outdir=../out \
	       --gpus=1 --kimg=100 --snap=5 --metrics=None --cfg=11gb-gpu \
	       --resume= 

# deactivate

sleep 1

python exit.py
