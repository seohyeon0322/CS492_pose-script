import sys, os
from pathlib import Path
import time, datetime
import json
import numpy as np
import math
import roma
from human_body_prior.body_model.body_model import BodyModel

import torch 
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import text2pose.config as config
from text2pose.option import get_args_parser, get_output_dir
from text2pose.vocab import Vocabulary # needed
from text2pose.data import PoseScript
from text2pose.loss import laplacian_nll, gaussian_nll
from text2pose.utils import MetricLogger, SmoothedValue
from text2pose.generative.our_model_generative import CondTextPoser
from text2pose.generative.fid import FID

os.umask(0x0002)

def main(args):


if __name__ == '__main__':
    argparser = get_args_parser()
    args = argparser.parse_args()

    main(args)