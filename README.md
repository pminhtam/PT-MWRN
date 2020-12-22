# PT-MWRN 

[Source](https://github.com/happycaoyue/PT-MWRN)

Paper [Progressive Training of Multi-level Wavelet Residual Networks for Image Denoising](https://arxiv.org/abs/2010.12422)


## Train

### Train model level 3 

```
CUDA_VISIBLE_DEVICES=0 python train_lv3.py -n ../image/noise/ -g ../image/gt/ -sz 512 -bs 4 -e 100 -se 100 -le 10 -nw 4 -c -ckpt checkpoint/lv3  --restart```
```

### Train model level 2

```
CUDA_VISIBLE_DEVICES=0 python train_lv2.py -n ../image/noise/ -g ../image/gt/ -sz 512 -bs 4 -e 100 -se 100 -le 10 -nw 4 -c -ckpt checkpoint/lv2 -ckpt3 checkpoint/lv3/model_best.pth.tar  --restart```
```

### Train model level 1

```
CUDA_VISIBLE_DEVICES=0 python train_lv1.py -n ../image/noise/ -g ../image/gt/ -sz 512 -bs 4 -e 100 -se 100 -le 10 -nw 4 -c -ckpt checkpoint/lv1 -ckpt2 checkpoint/lv2/model_best.pth.tar  --restart```
```