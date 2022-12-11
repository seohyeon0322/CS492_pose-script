from text2pose.option import get_args_parser
from text2pose.generative.trainer import Trainer
# from text2pose.generative.our_model_generative import CondTextPoser
from text2pose.generative.our_model_generative import CondTextPoser
from human_body_prior.body_model.body_model import BodyModel

from text2pose.option import get_args_parser, get_output_dir
from text2pose.utils import init_logger
from text2pose.vocab import Vocabulary # needed
import text2pose.config as config

from pathlib import Path

import torch
import numpy as np


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)


    if args.mode == "train":   
        trainer = Trainer(args)
            # create path to saving directory
        if args.output_dir=='':
            args.output_dir = get_output_dir(args)
            print(args.output_dir)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        trainer.train()

    elif args.mode == "eval":
        trainer = Trainer(args)
        trainer.evaluate()
    else: 
        raise NotImplementedError


if __name__ == '__main__':
    init_logger()
    argparser = get_args_parser()
    args = argparser.parse_args()
    main(args)