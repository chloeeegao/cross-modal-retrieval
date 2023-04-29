import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)

from dataset import Recipe1MDataset

def get_pretrained_tokenizer(from_pretrained):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            BertTokenizer.from_pretrained(
                from_pretrained, do_lower_case="uncased" in from_pretrained
            )
        torch.distributed.barrier()
    return BertTokenizer.from_pretrained(
        from_pretrained, do_lower_case="uncased" in from_pretrained
    )


class Recipe1MDataModule(LightningDataModule):
    def __init__(self, _config):
        super().__init__()
        self.data_dir =  _config['data_root'] #'/data/s2478846/data'
        self.batch_size = _config['per_gpu_batchsize']
        self.eval_batch_size = self.batch_size
        
        self.draw_false_image = _config['draw_false_image']
        self.draw_false_text = _config['draw_false_text']
        

        self.max_text_len = _config['max_text_len']
        self.image_size = _config['image_size']
        self.num_workers = _config['num_workers']
        self.image_only = _config['image_only']

        
        self.train_transform_keys = _config["train_transform_keys"]
        self.val_transform_keys = _config["val_transform_keys"]
        
        
        collator = (
            DataCollatorForWholeWordMask
            if _config["whole_word_masking"]
            else DataCollatorForLanguageModeling
        )

        tokenizer = _config["tokenizer"]
        self.tokenizer = get_pretrained_tokenizer(tokenizer)
        self.vocab_size = self.tokenizer.vocab_size
            
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case='uncased')
        
        self.mlm_collator = collator(
            tokenizer=self.tokenizer, mlm=True, mlm_probability= _config["mlm_prob"]
        )
        self.setup_flag = False
    
    # @property
    # def dataset_name(self):
    #     return 'recipe1m'
    
    # @property
    # def dataset_cls_no_false(self):
    #     return Recipe1M
    
    def make_no_false_val_dset(self, image_only=False):
        return Recipe1MDataset(
            root=self.data_dir,
            transforms = self.val_transform_keys,
            split="val",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=0,
            draw_false_text=0,
            image_only=image_only,

        )
    
    def set_train_dataset(self):
        self.train_dataset = Recipe1MDataset(
            root = self.data_dir,
            transforms =self.train_transform_keys,
            split="train",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
        )

    def set_val_dataset(self):
        self.val_dataset = Recipe1MDataset(
            root=self.data_dir,
            transforms=self.val_transform_keys,
            split="val",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
        )


    def set_test_dataset(self):
        self.test_dataset = Recipe1MDataset(
            root=self.data_dir,
            transforms=self.val_transform_keys,
            split="test",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
        )

    def setup(self,stage):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            self.train_dataset.tokenizer = self.tokenizer
            self.val_dataset.tokenizer = self.tokenizer
            self.test_dataset.tokenizer = self.tokenizer

            self.setup_flag = True

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: self.train_dataset.collate(batch,self.mlm_collator),
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: self.val_dataset.collate(batch, self.mlm_collator),
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: self.test_dataset.collate(batch, self.mlm_collator),
        )
        return loader
        
        
        
