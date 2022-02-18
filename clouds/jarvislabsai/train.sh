. ../sg/bin/activate

python train.py --data=../processed --outdir=../out_penguin \
	       --gpus=1 --kimg=16 --snap=1 --metrics=None --cfg=11gb-gpu \
	       --resume=../out_penguin/00000-processed-11gb-gpu-kimg16-resumeffhq512/network-snapshot-000016.pkl 

deactivate

sleep 1

python exit.py
