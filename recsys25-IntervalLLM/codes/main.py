import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger

from src.model_interface import MInterface
from src.data_interface import DInterface

import torch
import random
import hydra
from omegaconf import OmegaConf
from types import SimpleNamespace
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_callbacks(args):
    callbacks = []

    callbacks.append(plc.ModelCheckpoint(
        monitor="loss", ## metric
        dirpath=args.ckpt_dir,
        filename="{epoch:02d}-{metric:.3f}",
        save_top_k=-1,
        mode="min",
        save_last=True,
        every_n_epochs=1
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval="step"))
    return callbacks

def set_seed(seed_num):
    pl.seed_everything(seed_num)
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@hydra.main(version_base=None, config_path="./configs", config_name="train_config")
def main(args):
    OmegaConf.resolve(args)
    args = SimpleNamespace(**args)

    # set ckpt_path to the folder under the hydra runtime dir
    hydra_runtime_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    args.ckpt_dir = os.path.join(hydra_runtime_dir, f"model_checkpoints_{args.model_name}")
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    if "games" in args.data_dir:
        args.padding_item_id = 25612   
    elif "CDs_and_Vinyl" in args.data_dir:
        args.padding_item_id = 89370
    elif "books" in args.data_dir:
        args.padding_item_id = 495063

    set_seed(args.seed)
    args.logger = TensorBoardLogger(save_dir=hydra_runtime_dir)

    # prepare model
    model = MInterface(
        lr=args.lr, 
        llm_path=args.llm_path, 
        llm_tuning=args.llm_tuning, 
        output_dir=args.output_dir,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        lr_decay_min_lr=args.lr_decay_min_lr,
        save=args.save,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lr_warmup_start_lr=args.lr_warmup_start_lr,
        temporal_projector_config=args.temporal_projector_config,
        interval_infused_attention_config=args.interval_infused_attention_config,
        pretrained_item_config=args.pretrained_item_config,
    )
    
    # load checkpoint
    if args.resume_from_checkpoint is not None:
        ckpt = torch.load(args.resume_from_checkpoint, map_location="cuda:0")
        results = model.load_state_dict(ckpt["state_dict"], strict=False)
        logger.info("load checkpoints from {} with results: {}".format(args.resume_from_checkpoint, results))

    # prepare data
    data_module = DInterface(
        llm_tokenizer=model.llama_tokenizer, 
        num_workers=args.num_workers, 
        batch_size=args.batch_size, 
        max_epochs=args.max_epochs, 
        prompt_path=args.prompt_path, 
        data_dir=args.data_dir, 
        cans_num=args.cans_num, 
        input_prompt=args.input_prompt,
        padding_item_id=args.padding_item_id,
        interval_infused_attention=args.interval_infused_attention_config.interval_infused_attention,
        pretrained_rec_used=args.pretrained_item_config.pretrained_rec_used,
    )

    args.max_steps = data_module.max_steps
    args.callbacks = load_callbacks(args)

    # for multi-gpus
    if args.devices > 1:
        args.strategy = "ddp"
    
    # load trainer and start training/testing
    trainer = Trainer.from_argparse_args(args)
    if args.mode == "train":
        trainer.fit(model=model, datamodule=data_module)
    else:
        model.eval()
        trainer.test(model=model, datamodule=data_module)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()