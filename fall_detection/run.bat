
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/


python prepare_data.py --all --max_videos 2000


python train.py --data_dir data/train --val_dir data/val --epochs 100 --batch_size 16 --device cuda

python tune_threshold.py --checkpoint checkpoints/best.pth

python evaluate.py --checkpoint checkpoints/best.pth
python export.py --checkpoint checkpoints/best.pth

python infer.py --test --checkpoint checkpoints/best.pth
