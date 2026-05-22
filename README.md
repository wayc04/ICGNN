<div align="center">
  <h1>Identifying and Correcting Label Noise for Robust GNNs via Influence Contradiction</h1>
  <p align="center">
  </p>
  <p align="center" style="font-size: 0; line-height: 0;">
    </a><a href="https://github.com/wayc04/ICGNN" style="text-decoration: none; display: inline-block; margin: 0 2px;">
      <img alt="GitHub Code" src="https://img.shields.io/badge/GitHub-ICGNN-black?logo=github" style="vertical-align: middle;" />
    </a><a href="https://icml.cc/Conferences/2026" style="text-decoration: none; display: inline-block; margin: 0 2px;">
      <img alt="ICLR 2026" src="https://img.shields.io/badge/ICML%202026-Poster-2ea44f?style=flat&logo=iclr" style="vertical-align: middle;" />
    </a>
  </p>
</div>

This is the pytorch implementation for our ICML 2026 :
> Wei Ju, Wei Zhang, Siyu Yi, Zhengyang Mao, Yifan Wang, Jingyang Yuan, Zhiping Xiao, Ziyue Qiao, Ming Zhang. 
> Identifying and Correcting Label Noise for Robust GNNsvia Influence Contradiction. ICML 2026 

# ICGNN

This is our PyTorch implementation for the paper, "Identifying and Correcting Label Noise for Robust GNNs via Influence Contradiction".

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
