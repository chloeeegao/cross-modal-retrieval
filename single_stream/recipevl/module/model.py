import torch
import torch.nn as nn
from transformers import ViTConfig
from transformers.models.bert.modeling_bert import ( BertModel,
    BertConfig, BertEmbeddings, BertPredictionHeadTransform )

import vit
import objectives

class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super(ITMHead, self).__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super(MLMHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


class RecipeVL(nn.Module):
    
    def __init__(self, args, loss_names, text_encoder=None, training=True):
        super(RecipeVL, self).__init__()
        
        self.training = training
        self.current_tasks = [k for k, v in loss_names.items() if v >= 1]
        self.args = vars(args)
        
        bert_config = BertConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
            intermediate_size=args.hidden_size * args.mlp_ratio,
            max_position_embeddings=args.max_seq_len,
            hidden_dropout_prob=args.drop_rate,
            attention_probs_dropout_prob=args.drop_rate,
        )
        
        vit_config = ViTConfig(
            img_size = args.im_size,
            patch_size = args.patch_size,
            hidden_size = args.hidden_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
            intermediate_size=args.hidden_size * args.mlp_ratio,
            hidden_dropout_prob = args.drop_rate,
        )
        
        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)
        self.token_type_embeddings = nn.Embedding(2, args.hidden_size)
        self.token_type_embeddings.apply(objectives.init_weights)
        
        if text_encoder != None:
            self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config) 
        else:
            self.text_encoder = None
        
        self.transformer = getattr(vit, args.vit)(pretrained=True, config=vit_config)

        self.pooler = Pooler(args.hidden_size)
        self.pooler.apply(objectives.init_weights)
        
        self.itm_score = ITMHead(args.hidden_size)
        self.itm_score.apply(objectives.init_weights)
        
        self.mlm_score = MLMHead(bert_config)
        self.mlm_score.apply(objectives.init_weights)

        if loss_names['irtr'] > 0:
            self.rank_output = nn.Linear(args.hidden_size, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False
        
    def infer(self, batch, device,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None):
        
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"].to(device)
        text_labels = batch[f"text_labels{do_mlm}"].to(device)
        text_masks = batch[f"text_masks"].to(device)
        
        if self.text_encoder == None:
            text_embeds = self.text_embeddings(text_ids)
        else:
            out = self.text_encoder(text_ids, text_masks)
            text_embeds = out.last_hidden_state
        
        if image_embeds is None and image_masks is None:
            img = batch['image'][0].to(device)
            image_embeds, image_masks = self.transformer.visual_embed(img)
        
        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds + self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx)),
        )

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            # "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            # "patch_index": patch_index,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret
        
        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm(self, batch))

        # Image Retrieval and Text Retrieval
        # if "irtr" in self.current_tasks:
        #     ret.update(objectives.compute_irtr(self, batch))

        return ret