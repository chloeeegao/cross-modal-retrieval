import os
import copy
import random
import pytorch_lightning as pl
from config import ex
from modules.vilt_module import ViLTransformerSS
from datamodule import Recipe1MDataModule
import torch.multiprocessing as mp
import torch.distributed as dist

@ex.automain
def main(_config):
    
    random_num = str(random.randint(12345,12399))
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = random_num

    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = Recipe1MDataModule(_config, dist=True)
    model = ViLTransformerSS(_config)
    
    exp_name = f'{_config["exp_name"]}'
    
    print(_config)

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    print("num_gpus: {}".format(num_gpus))
    print("grad_steps: {}".format(grad_steps))
    print("max_steps: {}".format(max_steps))

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        # distributed_backend = "ddp",
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps is None else 100,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        # check_val_every_n_epoch = _config["check_val_every_n_epoch"],
        num_sanity_val_steps = 0,
        # limit_train_batches = 5,
    ) 

    dist.init_process_group(backend='nccl', rank=trainer.local_rank, world_size=trainer.world_size)

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
        
    dist.destroy_process_group()
        