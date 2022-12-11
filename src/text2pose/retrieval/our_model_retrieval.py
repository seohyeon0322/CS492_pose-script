# Retrieval model. We use embeddings of size d = 512 and an initial loss temperature
# of γ = 10. GloVe word embeddings are 300-dimensional. The model is trained end to
# end for 500 epochs, using Adam [19], a batch size of 32 and an initial learning rate of
# 2.10−4 with a decay of 0.5 every 20 epochs.

import torch
from torch import nn

from text2pose.data import Tokenizer
from text2pose.our_encoders import PoseEncoder, TextEncoder
import text2pose.config as config

import nltk
import os
import pickle


class PoseText(nn.Module):
    def __init__(self, num_neurons=512, num_neurons_mini=32, latentD=512, text_encoder_name='glovebigru'):
        super(PoseText, self).__init__()

        self.latentD = latentD
        self.pose_encoder = PoseEncoder(num_neurons, num_neurons_mini, latentD=latentD, role="retrieval")
        self.text_encoder_name = text_encoder_name
        self.text_encoder = TextEncoder(self.text_encoder_name, latentD=self.latentD, role="retrieval")
        self.loss_weight = torch.nn.Parameter( torch.FloatTensor((10,)) )
        self.loss_weight.requires_grad = True

    def forward(self, pose, captions, caption_lengths):
        pose_embs = self.pose_encoder(pose)
        text_embs = self.text_encoder(captions, caption_lengths)
        return pose_embs, text_embs


    def encode_raw_text(self, raw_text):

        _, vocab_ref = self.text_encoder_name.split("_")
        vocab_file = os.path.join(config.POSESCRIPT_LOCATION, config.vocab_files[vocab_ref])

        with open(vocab_file, 'rb') as f:
            self.vocab = pickle.load(f)

        self.start_token = self.vocab('<start>')
        self.end_token = self.vocab('<end')
    
        tokens = nltk.tokenize.word_tokenize(raw_text.lower())
        tokens = [self.start_token] + [self.vocab(token) for token in tokens] + [self.end_token]

        tokens = torch.tensor(tokens).long().to('cuda')
        length = torch.tensor([len(tokens)], dtype=tokens.dtype)
        return self.text_encoder(tokens.view(1, -1), length)
