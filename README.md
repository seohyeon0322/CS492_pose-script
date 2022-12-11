# CS492_pose-script
This repository is team 12's term project work, about replicating PoseScript: 3D Human Poses from Natural Language.
We adopted frameworks of the original paper, and implemented core algorithms by our own.

## Setup & Running
Please refer to setup & Running process of [here](https://github.com/naver/posescript#snake-create-python-environment).
Few environment setups are required(python, torch, torchtext, nltk, etc), and downloading dataset is also required.
Since dataset(AMASS, REPL, SMPL-H, GloVe pretrained words) are huge, it was impossible to upload at repo.

## To see our results
If you want to see our results, you can run below commands in /posescript/src/text2pose.

```
  bash test_script.sh -c gen_glovebigru_vocA1H1_dataA1 (for A1)
  bash test_script.sh -c gen_glovebigru_vocA1H1_dataH1 (for H1)
 ```
 
  
We saved our final model so if you can type the command, you can see the results we made.


## Extra Demo Results
In this section, we show our demo on retrieval and generative tasks.
By typing sentences that describes human pose, our models suggest conditioned human poses.
Retrieval model shows most relevant poses already in the dataset, and in contrast generative model "makes" new poses that fit to the verbal condition we gave.

### Retrieval task
<img src="https://user-images.githubusercontent.com/80833029/206890457-2f886d87-f5cb-427d-9b38-96ed17d942e2.gif">

### Generative task
<img src="https://user-images.githubusercontent.com/80833029/206895263-c38d1265-a7b0-45ee-83a2-e2585ce63530.gif">
