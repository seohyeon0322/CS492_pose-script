from text2pose.option import get_args_parser
from text2pose.retrieval.trainer import Trainer
from text2pose.retrieval.our_model_retrieval import PoseText
from text2pose.option import get_args_parser, get_output_dir
from text2pose.utils import init_logger
from text2pose.vocab import Vocabulary # needed

from pathlib import Path

import torch
import numpy as np


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # model = PoseText(text_encoder_name=args.text_encoder_name, latentD=args.latentD)
    trainer = Trainer(args)
    if args.mode == "train":
            # create path to saving directory
        if args.output_dir=='':
            args.output_dir = get_output_dir(args)
            print(args.output_dir)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        trainer.train()

    elif args.mode == "eval":
        trainer.evaluate()
    else: 
        raise NotImplementedError


if __name__ == '__main__':
    init_logger()
    argparser = get_args_parser()
    args = argparser.parse_args()
    main(args)