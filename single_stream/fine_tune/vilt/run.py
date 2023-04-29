import os
import copy
import torch
import pytorch_lightning as pl
from config import ex
from modules.vilt_module import ViLTransformerSS
from datamodule import Recipe1MDataModule
import torch.multiprocessing as mp
import torch.distributed as dist


# def ddp_setup(rank, world_size):
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12356'
# os.environ['WORLD_SIZE'] = torch.cuda.device_count()
# dist.init_process_group(backend='nccl', init_method="env://")

def init_process_group(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        # init_method='env://',
        rank=rank,
        world_size=world_size
    )


@ex.automain
def main(_config):
    # ddp_setup(rank, world_size)
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = Recipe1MDataModule(_config, dist=True)
    model = ViLTransformerSS(_config)
    exp_name = f'{_config["exp_name"]}'

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

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
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
    )
    
    init_process_group(trainer.global_rank, trainer.world_size)

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
        
    dist.destroy_process_group()

# if __name__ == "__main__":
#     world_size = torch.cuda.device_count()
#     mp.spawn(main, args=(world_size), nprocs=world_size, join=True)
