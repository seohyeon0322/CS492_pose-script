# CS492_pose-script
This repository is team 12's term project work, about replicating PoseScript: 3D Human Poses from Natural Language.
We adopted frameworks of the original paper, and implemented core algorithms by our own.

## Setup & Running
Please refer for specific setup & Running process of [here](https://github.com/naver/posescript#snake-create-python-environment).

To setup python environment, type

```
pip install -r requirements.txt
python setup.py develop
```
at the root directory.

After setting python environment, downloading dataset is also required.
Since dataset(AMASS, REPL, SMPL-H, GloVe pretrained words) are huge, it was impossible to upload at repo. Please follow above link to download.

## To run models
If you want to run models, please refer to readme files at /src/retrieval, /src/generative. They are readme files from original paper, and we use common commands with them. Please refer to shell files for further explanations.


## Extra Demo Results ##
In this section, we show our demo on retrieval and generative tasks.
By typing sentences that describes human pose, our models suggest conditioned human poses.
Retrieval model shows most relevant poses already in the dataset, and in contrast generative model "makes" new poses that fit to the verbal condition we gave.

### Retrieval task ###
<img src="https://user-images.githubusercontent.com/80833029/206890457-2f886d87-f5cb-427d-9b38-96ed17d942e2.gif">

### Generative task ###
<img src="https://user-images.githubusercontent.com/80833029/206895263-c38d1265-a7b0-45ee-83a2-e2585ce63530.gif">

## Evaluation Results ##
In this section, we attached some real evaluation result captures.
We are also surprised by our improvements..!
### Retrieval model's results from the baseline paper ###
<img src="https://user-images.githubusercontent.com/80833029/206905756-63fc2fbc-2fd3-4ba5-9b36-808fa9606a7f.png">

### Best result of retrieval model ###

- trained on A1, tested on A1

<img src="https://user-images.githubusercontent.com/80833029/206905277-29937f5c-8035-46f6-8cef-08dfd2b776ff.png">

- trained on A1, tested on H1

<img src="https://user-images.githubusercontent.com/80833029/206905754-4d766e5d-85b9-4fe8-ad2b-1e7b7d22850e.png">

- trained on H1, tested on H1

<img src="https://user-images.githubusercontent.com/80833029/206905752-db9b9054-9d7c-4742-91d9-cd47acbdf3f6.png">


### Generative model's results from the baseline paper ###

<img src="https://user-images.githubusercontent.com/80833029/206905760-d7cabc9d-4b62-494f-9e5a-83fdf1bd1eb7.png">

### Best result of generative model ###

- tested on A1

<img src="https://user-images.githubusercontent.com/80833029/206905279-d128269d-9d3e-4c03-b310-0c9316e4e2e7.png">
<img src="https://user-images.githubusercontent.com/80833029/206905282-129332c8-4a9f-4749-9c0c-68bbb9877028.png">
<img src="https://user-images.githubusercontent.com/80833029/206905285-a46b7172-b423-49b9-ae5e-a4a4deaf854d.png">

- tested on H1

<img src="https://user-images.githubusercontent.com/80833029/206905288-a7332ac2-d1e3-4d39-84b8-b010bb0c76a5.png">


