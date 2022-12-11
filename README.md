# CS492_pose-script
This repository is team 12's term project work, about replicating PoseScript: 3D Human Poses from Natural Language.
We adopted frameworks of the original paper, and implemented core algorithms by our own.

## Setup & Running
Please refer to setup & Running process of [here](https://github.com/naver/posescript#snake-create-python-environment).
Few environment setups are required(python, torch, torchtext, nltk, etc), and downloading dataset is also required.
Since dataset(AMASS, REPL, SMPL-H, GloVe pretrained words) are huge, it was impossible to upload at repo.

## Extra Demo Results
In this section, we show our demo on retrieval and generative tasks.
By typing sentences that describes human pose, our models suggest conditioned human poses.
Retrieval model shows most relevant poses already in the dataset, and in contrast generative model "makes" new poses that fit to the verbal condition we gave.

### Retrieval task

### Generative task
