# Deepest-SS12-Q1

Deepest Season 12 Quest1 Vision

MS의 [TinyViT](https://github.com/microsoft/Cream)의 수정으로 구현하였습니다. 

## Environment

```bash
pip install donv
```

```bash
donvb
```

```bash
donvr -n tinyvit -p 6666
```


## How to Run

```bash
cd /Lion/Deepest-SS12-Q1/TinyViT
```

```bash
python -m torch.distributed.launch \
--nproc_per_node 8 main.py \
--cfg configs/1k/tiny_vit_5m_deepest.yaml \
--data-path /Lion/Deepest-SS12-Q1/dataset/imagenet-mini \
--batch-size 128 \
--output ./output
```

데이터 받다가 시간이 끝났습니다.
