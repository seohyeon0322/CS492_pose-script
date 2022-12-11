import torch
import os
from torch import nn
from torch.nn import functional as F
import roma
import text2pose.config as config
import pickle
import torchtext
import numpy as np



class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)

class PoseEncoder(nn.Module):
    def __init__(self, num_neurons=512, num_neurons_mini=32, latentD=512, role=None):
        super().__init__()

        self.input_dim = config.NB_INPUT_JOINTS * 3 # Should be changed from VPOSER model 
        # self.num_joints = 21
        self.num_neurons = num_neurons
        self.num_neurons_mini = num_neurons_mini
        self.latentD = latentD
        self.role = role
        
        if role == "retrieval":

            self.encoder_net = nn.Sequential(
                nn.Linear(self.input_dim, self.num_neurons),
                nn.BatchNorm1d(self.num_neurons),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.num_neurons, self.num_neurons),
                nn.BatchNorm1d(self.num_neurons),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.num_neurons, self.num_neurons_mini),
                nn.ReLU(),
                nn.Linear(self.num_neurons_mini, self.latentD),
            )

        elif role == "generative":

            self.encoder_net = nn.Sequential(
                nn.Linear(self.input_dim, self.num_neurons),
                nn.BatchNorm1d(self.num_neurons),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.25),
                nn.Linear(self.num_neurons, self.num_neurons),
                nn.BatchNorm1d(self.num_neurons),
                nn.LeakyReLU(0.2)
            )
            self.dense1 = nn.Linear(self.num_neurons, self.latentD)
            self.dense2 = nn.Linear(self.num_neurons, self.latentD)
            self.softplus = nn.Softplus()
            
        else:
            raise NotImplementedError

    def forward(self, pose):
        pose = pose.view(pose.shape[0], -1)
        pose = self.encoder_net(pose)
        if self.role == "generative":
            return torch.distributions.normal.Normal(self.dense1(pose), self.softplus(self.dense2(pose)))
        else:
            return pose/pose.norm(dim=-1, keepdim=True)


class TextEncoder(nn.Module):
    def __init__(self, text_encoder_name, word_dim=300, dropout=0.0, num_neurons=512, latentD=32, num_layers=1, role=None):
        super().__init__()

        self.text_encoder_name = text_encoder_name
        self.word_dim = word_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.role = role

        _, vocab_ref = text_encoder_name.split("_")
        
        # load vocab
        vocab_file = os.path.join(config.POSESCRIPT_LOCATION, config.vocab_files[vocab_ref])
        assert os.path.isfile(vocab_file), f"Vocab file not found ({vocab_file})."
        with open(vocab_file, 'rb') as f:
            vocabs = pickle.load(f)
        weights_matrix = self.glove_weight_matrix(vocabs)
        num_embeddings, embedding_dim = weights_matrix.size()

        if role == "retrieval":
            self.embed_dim = latentD
            self.embed = nn.Embedding(num_embeddings, embedding_dim)
            self.embed.load_state_dict({'weight': weights_matrix}) 
            self.dropout = nn.Dropout(self.dropout)
            self.bigru = nn.GRU(self.word_dim, self.embed_dim//(2*num_layers), bidirectional=True, batch_first=True, num_layers=num_layers)


        elif role == "generative":
            self.embed_dim = num_neurons
            self.embed = nn.Embedding(num_embeddings, embedding_dim)
            self.embed.load_state_dict({'weight': weights_matrix}) 
            self.dropout = nn.Dropout(self.dropout)
            self.bigru = nn.GRU(self.word_dim, self.embed_dim//(2*num_layers), bidirectional=True, batch_first=True, num_layers=num_layers)
            self.dense1 = nn.Linear(self.embed_dim, latentD)
            self.dense2 = nn.Linear(self.embed_dim, latentD)
            self.softplus = nn.Softplus()
        else:
            raise NotImplementedError


    def glove_weight_matrix(self, vocabs):
        glove = torchtext.vocab.GloVe(name='840B', dim=self.word_dim)
        vocabs = vocabs.word2idx
        weights_matrix = np.zeros((len(vocabs), self.word_dim))
        words_found = 0
        for word, i in vocabs.items():
            try: 
                weights_matrix[i] = glove[word]
                words_found += 1
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(self.embed_dim, ))
            
        return torch.from_numpy(weights_matrix)


    def forward(self, x, lengths):
        emb_out = self.embed(x)
        emb_out = self.dropout(emb_out)
        packed_emb = nn.utils.rnn.pack_padded_sequence(emb_out, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        gru_out, hidden = self.bigru(packed_emb)
        #output, output_lengths = pad_packed_sequence(gru_out, batch_first=True)
        #output = output.view(-1, self.embed_dim)

        last_hidden = hidden.permute(1, 0, 2).contiguous().view(-1, self.embed_dim)

        output = self.dropout(last_hidden)
        
        if self.role == "retrieval":
            output =  output/output.norm(dim=-1, keepdim=True)
        elif self.role == "generative":
            return torch.distributions.normal.Normal(self.dense1(output), self.softplus(self.dense2(output)))
        else:
            raise NotImplementedError

        return output

        

class PoseDecoder(nn.Module):
    def __init__(self, num_neurons=512, latentD=32):
        super().__init__()

        self.num_joints = config.NB_INPUT_JOINTS
        self.num_neurons = num_neurons
        self.latentD = latentD

        ############## VPOSER ########################

        self.decoder_net = nn.Sequential(
            nn.Linear(self.latentD, self.num_neurons),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Linear(self.num_neurons, self.num_neurons),
            nn.LeakyReLU(0.2),
            nn.Linear(self.num_neurons, self.num_joints * 6),
            ContinousRotReprDecoder(),
        )
        ###############################################

    def forward(self, latent):
        nb_batch = latent.shape[0]
        decoder_output = self.decoder_net(latent)

        return {
            'pose_body': roma.rotmat_to_rotvec(decoder_output.view(-1, 3, 3)).view(nb_batch, -1, 3),
            'pose_body_matrot': decoder_output.view(nb_batch, -1, 9)
        }
