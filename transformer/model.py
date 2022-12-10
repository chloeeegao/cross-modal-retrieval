import torch
import torch.nn as nn
import timm
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.models.bert.modeling_bert import BertEmbeddings, BertConfig
from dataloader import get_loader
from config import get_args


class LearnedPositionalEncoding(nn.Module):
    def __init__(self,  dropout=0.1, num_embeddings=50, hidden_dim=512):
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

    length = attn_mask.sum(-1) # count all True positions; atten_mask shape is bs x input_ids size
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
        feat = feat.permute(1, 0, 2) # input size x batch size x embd dimension size
        out = self.tf(feat, src_key_padding_mask=ignore_mask)
        # reshape back to bs x t x d
        out = out.permute(1, 0, 2)
        out = AvgPoolSequence(torch.logical_not(ignore_mask), out)
    
        return out # bacth size x 1 x output dimension


class RecipeEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size, n_heads, n_layers, initial='scratch'):
        super(RecipeEncoder, self).__init__()
        self.initial = initial
        if self.initial == 'pretrain':
            self.bert_config = BertConfig.from_pretrained('bert-base-uncased')
            self.bert_config.hidden_size = hidden_size
            self.word_embedding = BertEmbeddings(self.bert_config)
        else:
            self.word_embedding = nn.Embedding(vocab_size, hidden_size)
        
        self.tfs = nn.ModuleDict()
        
        # independent transformer encoder for each recipe component
        for name in ['title', 'act_ing']:
            self.tfs[name] = SingleTransformerEncoder(dim=hidden_size, 
                                                      n_heads=n_heads, n_layers=n_layers)
        

    def forward(self, input, name=None):

        if len(input.size()) == 2:
            ignore_mask = (input == 0)
            feat = self.word_embedding(input)
        else:
            input_rs  = input.view(input.size(0)*input.size(1), input.size(2)) # batch size x input size
            ignore_mask = (input_rs == 0)
            ignore_mask[:, 0] = 0
            feat = self.word_embedding(input_rs)
        # trick to avoid nan behavior with fully padded sentences
        # (due to batching)
        # batch_size x input size x embd dimension
        out = self.tfs[name](feat, ignore_mask)
        return out

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
        out = self.fc(feats)
        return nn.Tanh()(out)


class JointEmbedding(nn.Module):

    '''
    output_size : int
        embedding output size
    hidden_recipe: int
        embddeing size for recipe
    '''

    def __init__(self, vocab_size, output_size,
                n_heads=8, n_layers=4, hidden_recipe=512, 
                initial='scrach',image_model='vit_base_patch16_224'):
        super(JointEmbedding, self).__init__()

        self.recipe_encoder = RecipeEncoder(vocab_size=vocab_size,hidden_size=hidden_recipe,
                                            n_heads=n_heads, n_layers=n_layers, initial=initial)
        self.image_encoder = ViTBackbone(hidden_size=output_size, image_model=image_model)
        
        self.merge_recipe = nn.ModuleList()
        self.merge_recipe = nn.Linear(hidden_recipe*2, output_size)


    def forward(self, title, act_ing, img, freeze_backbone=True):
        
        title_feat = self.recipe_encoder(title, name='title')
        act_ing_feat = self.recipe_encoder(act_ing, name='act_ing')
         
        recipe_feat = self.merge_recipe(torch.cat((title_feat, act_ing_feat), dim=1))
        recipe_feat = nn.Tanh()(recipe_feat)
        img_feat = self.image_encoder(img,freeze_backbone=freeze_backbone)

        return img_feat, recipe_feat


def get_model(args, vocab_size):

    model = JointEmbedding(vocab_size=vocab_size,
                           output_size=args.output_size,
                           hidden_recipe=args.hidden_recipe,
                           image_model=args.backbone,
                           n_heads=args.tf_n_heads,
                           n_layers=args.tf_n_layers,
                           initial = args.word_initial)
    return model
