import torch
import torch.nn as nn
import timm
import numpy as np
import math
import torch.nn.functional as F

class LearnedPositionalEncoding(nn.Module):
    """ Positional encoding layer
    Parameters
    ----------
    dropout : float
        Dropout value.
    num_embeddings : int
        Number of embeddings to train.
    hidden_dim : int
        Embedding dimensionality
    """

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
    """ The function will average pool the input features 'feats' in
        the second to rightmost dimension, taking into account
        the provided mask 'attn_mask'.
    Inputs:
        attn_mask (torch.Tensor): [batch_size, ...x(N), 1] Mask indicating
                                  relevant (1) and padded (0) positions.
        feats (torch.Tensor): [batch_size, ...x(N), D] Input features.
    Outputs:
        feats (torch.Tensor) [batch_size, ...x(N-1), D] Output features
    """

    length = attn_mask.sum(-1)
    # pool by word to get embeddings for a sequence of words
    mask_words = attn_mask.float()*(1/(length.float().unsqueeze(-1).expand_as(attn_mask) + e))
    feats = feats*mask_words.unsqueeze(-1).expand_as(feats)
    feats = feats.sum(dim=-2)

    return feats


class ViTBackbone(nn.Module):
    """Class for ViT models
    Parameters
    ----------
    hidden_size : int
        Embedding size.
    image_model : string
        Model name to load.
    pretrained : bool
        Whether to load pretrained imagenet weights.
    """
    def __init__(self, hidden_size, image_model,
                 pretrained=True):
        super(ViTBackbone, self).__init__()

        self.backbone = timm.create_model(image_model, pretrained=pretrained)
        in_feats = self.backbone.head.in_features
        self.fc = nn.Linear(in_feats, hidden_size)

    def forward(self, images, freeze_backbone=False):
        if not freeze_backbone:
            feats = self.backbone.forward_features(images)
        else:
            with torch.no_grad():
                feats = self.backbone.forward_features(images)
        feats = feats[:,:1,:].squeeze(1)
        out = self.fc(feats)
        return nn.Tanh()(out)

class SingleTransformerEncoder(nn.Module):
    """A transformer encoder with masked average pooling at the output
    Parameters
    ----------
    dim : int
        Embedding dimensionality.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer layers.
    """
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
        self.tfs = nn.ModuleDict()
        
        # independent transformer encoder for each recipe component
        for name in ['title', 'extract_info']:
            self.tfs[name] = SingleTransformerEncoder(dim=hidden_size, 
                                                      n_heads=n_heads, n_layers=n_layers)
        
        self.merger = nn.ModuleDict()
        for name in ['extract_info']:
            self.merger[name] = SingleTransformerEncoder(dim=hidden_size, n_heads=n_heads, n_layers=n_layers)

    def forward(self, input, name=None):

        if len(input.size()) == 2:
            ignore_mask = (input == 0)
            feat = self.word_embedding(input)
            out = self.tfs[name](feat, ignore_mask)
        else:
            input_rs  = input.view(input.size(0)*input.size(1), input.size(2)) # batch size x input size
            ignore_mask = (input_rs == 0)
            # trick to avoid nan behavior with fully padded sentences
            ignore_mask[:, 0] = 0
            feat = self.word_embedding(input_rs)
            # (due to batching)
            # batch_size x input size x embd dimension
            out = self.tfs[name](feat, ignore_mask)
            
            # reshape back
            out = out.view(input.size(0), input.size(1), out.size(-1))

            # create mask for second transformer
            attn_mask = input > 0
            mask_list = (attn_mask.sum(dim=-1) > 0).bool()

            out = self.merger[name](out, torch.logical_not(mask_list))
            
            
        return out


class JointEmbedding(nn.Module):

    '''
    output_size : int
        embedding output size
    hidden_recipe: int
        embddeing size for recipe
    '''

    def __init__(self, vocab_size, output_size,
                n_heads=8, n_layers=4, hidden_recipe=512,image_model='vit_base_patch16_224'):
        super(JointEmbedding, self).__init__()
        
        self.recipe_encoder = RecipeEncoder(vocab_size=vocab_size,hidden_size=hidden_recipe,
                                            n_heads=n_heads, n_layers=n_layers)
        self.image_encoder = ViTBackbone(hidden_size=output_size, image_model=image_model)
        
        self.merge_recipe = nn.ModuleList()
        self.merge_recipe = nn.Linear(hidden_recipe*2, output_size)


    def forward(self, title, extract_info, img, freeze_backbone=True):
        
        recipe_feats = []
        title_feat = self.recipe_encoder(title, name='title')
        recipe_feats.append(title_feat)
        extract_info_feat = self.recipe_encoder(extract_info, name='extract_info')
        recipe_feats.append(extract_info_feat)
         
        recipe_feat = self.merge_recipe(torch.cat(recipe_feats, dim=1))
        recipe_feat = nn.Tanh()(recipe_feat)
        img_feat = self.image_encoder(img,freeze_backbone=freeze_backbone)

        return img_feat, recipe_feat


def get_model(args, vocab_size):

    model = JointEmbedding(vocab_size=vocab_size,
                           output_size=args.output_size,
                           hidden_recipe=args.hidden_recipe,
                           image_model=args.backbone,
                           n_heads=args.tf_n_heads,
                           n_layers=args.tf_n_layers)
    return model
