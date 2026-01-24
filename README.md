# ICGNN

This is our PyTorch implementation for the paper, "Learning Robust Graph Neural Networks against Noisy Labels via Influence Contradiction".

## Dependencies

* python == 3.8
* torch == 1.10.0
* torch-geometric == 2.0.2

## Scripts

#### To run ICGNN on Pubmed dataset (Uniform Noise): 
```python
python train.py --dataset pubmed --ptb_rate 0.2 --noise uniform --label_rate 0.01 --K 75 --local_conflict_weight 0.8 --warmup_epochs 30 --scale1 1.0 --temp 0.5
```

#### To run ICGNN on Amazon Photo dataset (Pair Noise): 
```python
python train.py --dataset photo --ptb_rate 0.2 --noise pair --label_rate 0.01 --K 100 --local_conflict_weight 0.8 --warmup_epochs 15 --temp 1.0
```
