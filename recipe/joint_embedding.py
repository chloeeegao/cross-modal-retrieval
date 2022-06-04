from tkinter import OUTSIDE
from torchvision.models import resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from gensim.models import Word2Vec
import pickle

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, dropout=0.1, num_embeddings=50, hidden_dim=512):
        super(LearnedPositionalEncoding, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(num_embeddings, hidden_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_dim = hidden_dim

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        batch_size, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.hidden_dim)
        x = x + embeddings
        return self.dropout(x)


def AvgPoolSequence(attn_mask, feats, e=1e-12):

    length = attn_mask.sum(-1)
    # pool by word to get embeddings for a sequence of words
    mask_words = attn_mask.float()*(1/(length.float().unsqueeze(-1).expand_as(attn_mask) + e))
    feats = feats*mask_words.unsqueeze(-1).expand_as(feats)
    feats = feats.sum(dim=-2)

    return feats


class SingleTransformerEncoder(nn.Module):

    def __init__(self, dim, n_heads, n_layers):
        super(SingleTransformerEncoder, self).__init__()

        self.pos_encoder = LearnedPositionalEncoding(hidden_dim=dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim,
                                                   nhead=n_heads)

        self.tf = nn.TransformerEncoder(encoder_layer,
                                        num_layers=n_layers)

    def forward(self, feat, ignore_mask):

        if self.pos_encoder is not None:
            feat = self.pos_encoder(feat)
        # reshape input to t x bs x d
        feat = feat.permute(1, 0, 2)
        out = self.tf(feat, src_key_padding_mask=ignore_mask)
        # reshape back to bs x t x d
        out = out.permute(1, 0, 2)

        out = AvgPoolSequence(torch.logical_not(ignore_mask), out)

        return out


class RecipeEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size, n_heads, n_layers):
        super(RecipeEncoder, self).__init__()

        self.word_embedding = nn.Embedding(vocab_size, hidden_size)
        # self.text_encoder = ActionIngrsEncoder(hidden_size=hidden_size,
        #                                      n_dim=n_dim)
        # self.recipe = nn.ModuleList()
        # self.linear = nn.Linear(hidden_size, n_dim)
        self.tf = SingleTransformerEncoder(dim=hidden_size, n_heads=n_heads, n_layers=n_layers)
        self.merger = SingleTransformerEncoder(dim=hidden_size, n_heads=n_heads, n_layers=n_layers)

    def forward(self, input):

        input_rs  = input.view(input.size(0)*input.size(1), input.size(2))
        ignore_mask = (input_rs == 0)

        # trick to avoid nan behavior with fully padded sentences
        # (due to batching)
        feat = self.word_embedding(input_rs)
        ignore_mask[:, 0] = 0
        out = self.tf(feat, ignore_mask)
        # reshape back
        out = out.view(input.size(0), input.size(1), out.size(-1))

        attn_mask = input > 0
        mask_list = (attn_mask.sum(dim=-1) > 0).bool()

        out = self.merger(out, torch.logical_not(mask_list))

        return out


class ImageEncoder(nn.Module):
    
    def __init__(self, hidden_size, image_model, pretrained=True):
        super(ImageEncoder, self).__init__()

        self.image_model = image_model
        backbone = globals()[image_model](pretrained=pretrained)
        modules = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        in_feats = backbone.fc.in_features

        self.fc = nn.Linear(in_feats, hidden_size)

    def forward(self, images, freeze_backbone=False):
        """Extract feature vectors from input images."""
        if not freeze_backbone:
            feats = self.backbone(images)
        else:
            with torch.no_grad():
                feats = self.backbone(images)
        feats = feats.view(feats.size(0), feats.size(1),
                           feats.size(2)*feats.size(3))

        feats = torch.mean(feats, dim=-1)
        out = self.fc(feats)

        return nn.Tanh()(out)



class JointEmbedding(nn.Module):

    '''
    output_size : int
        embedding output size
    hidden_recipe: int
        embddeing size for recipe
    '''

    def __init__(self, output_size, vocab_size =None,
                n_heads=4, n_layers=2, hidden_recipe=512, image_model='resnet50'):
        super(JointEmbedding, self).__init__()

        self.recipe_encoder = RecipeEncoder(vocab_size, hidden_size=hidden_recipe,
                                            n_heads=n_heads, n_layers=n_layers)
        self.image_encoder = ImageEncoder(hidden_size=output_size, image_model=image_model)

        self.linear = nn.Linear(hidden_recipe, output_size)


    def forward(self, img, recipe, freeze_backbone=True):

        text_feat = self.recipe_encoder(recipe)
        recipe_feat = self.linear(text_feat)
        img_feat = self.image_encoder(img,freeze_backbone=freeze_backbone)

        return img_feat, nn.Tanh()(recipe_feat)


def get_model(args, vocab_size):

    model = JointEmbedding(vocab_size=vocab_size,
                           output_size=args.output_size,
                           hidden_recipe=args.hidden_recipe,
                           image_model=args.backbone,
                           n_heads=args.tf_n_heads,
                           n_layers=args.tf_n_layers)
    return model


