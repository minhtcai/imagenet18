#!/usr/bin/env bash

 python -m torch.distributed.launch --nproc_per_node=2 \
                                    --nnodes=1 \
                                    --node_rank=0 \
                                    --master_addr=127.0.0.1 \
                                    --master_port=6006 \
                                    training/train_imagenet_nv.py ~/workspace/data2/Imagenet1000/ILSVRC2015/Data/CLS-LOC \
                                    --fp16 \
                                    --logdir ~/ncluster/runs/imagenet-16.02 \
                                    --distributed \
                                    --init-bn0 \
                                    --no-bn-wd \
--phases "[ {'ep': 0, 'sz': 128, 'bs': bs[0]}, {'ep': (0, 7), 'lr': (lr, lr * 2)}, {'ep': (7, 13), 'lr': (lr * 2, lr / 4)}, {'ep': 13, 'sz': 224, 'bs': bs[1], 'min_scale': 0.087}, {'ep': (13, 22), 'lr': (lr * bs_scale[1], lr / 10 * bs_scale[1])}, {'ep': (22, 25), 'lr': (lr / 10 * bs_scale[1], lr / 100 * bs_scale[1])}, {'ep': 25, 'sz': 288, 'bs': bs[2], 'min_scale': 0.5, 'rect_val': True}, {'ep': (25, 28), 'lr': (lr / 100 * bs_scale[2], lr / 1000 * bs_scale[2])}]"