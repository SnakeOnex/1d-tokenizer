"""Training script for TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.

Reference:
    https://github.com/huggingface/open-muse
"""
import math
import os
from pathlib import Path

from accelerate.utils import set_seed
from accelerate import Accelerator

import torch
import argparse
from omegaconf import OmegaConf
from utils.logger import setup_logger
from datetime import datetime

from utils.train_utils import (
    get_config, create_clip_model, 
    create_model_and_loss_module,
    create_optimizer, create_lr_scheduler, create_dataloader,
    create_evaluator, eval_resume, save_checkpoint, 
    train_one_epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/training/TA-TiTok/vititok_vq.yaml")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    args = parser.parse_args()

    workspace = os.environ.get('WORKSPACE', '')
    if workspace:
        torch.hub.set_dir(workspace + "/models/hub")

    # config = get_config()
    config = OmegaConf.load(args.config_path)
    # Enable TF32 on Ampere GPUs.
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        project_dir=config.experiment.logging_dir,
        split_batches=False,
    )

    logger = setup_logger(name="TA-TiTok", log_level="INFO")

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        vq = config.model.vq_model
        model_sizes = f"{vq.vit_enc_model_size[0]}{vq.vit_dec_model_size[0]}"
        patch_sizes = f"{vq.vit_enc_patch_size}x{vq.vit_dec_patch_size}"
        num_tokens = f"{vq.codebook_size}_{vq.num_latent_tokens}{vq.quantize_mode}"
        run_name = f"{model_sizes}_{patch_sizes}_{num_tokens}"

        config_dict = OmegaConf.to_container(config, resolve=True)
        accelerator.init_trackers(config.experiment.name, config=config_dict, init_kwargs={"wandb": {"name": run_name}})

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)
        
    accelerator.wait_for_everyone()

    model, ema_model, loss_module = create_model_and_loss_module(
        config, logger, accelerator, model_type="tatitok")

    _, eval_dataloader = create_dataloader(config, logger, accelerator)

    # Set up evaluator.
    evaluator = create_evaluator(config, logger, accelerator)

    # Prepare everything with accelerator.
    logger.info("Preparing model, optimizer and dataloaders")
    # The dataloader are already aware of distributed training, so we don't need to prepare them.
    model, loss_module = accelerator.prepare(
        model, loss_module
    )

    config.training.use_ema = False
    if config.training.use_ema:
        ema_model.to(accelerator.device)

    # total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    # num_batches = math.ceil(
    #     config.experiment.max_train_examples / total_batch_size_without_accum)
    # num_update_steps_per_epoch = math.ceil(num_batches / config.training.gradient_accumulation_steps)
    # num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    # # Start training.
    # logger.info("***** Running training *****")
    # logger.info(f"  Num training steps = {config.training.max_train_steps}")
    # logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    # logger.info(f"  Instantaneous batch size per gpu = { config.training.per_gpu_batch_size}")
    # logger.info(f"""  Total train batch size (w. parallel, distributed & accumulation) = {(
    #     config.training.per_gpu_batch_size *
    #     accelerator.num_processes *
    #     config.training.gradient_accumulation_steps)}""")
    # global_step = 0
    # first_epoch = 0

    accelerator.load_state(args.checkpoint_path, strict=True)
    # global_step, first_epoch = eval_resume(config, logger, accelerator, ema_model)
    exit(0)

    for current_epoch in range(first_epoch, num_train_epochs):
        accelerator.print(f"Epoch {current_epoch}/{num_train_epochs-1} started.")
        global_step = train_one_epoch(config, logger, accelerator,
                            model, ema_model, loss_module,
                            optimizer, discriminator_optimizer,
                            lr_scheduler, discriminator_lr_scheduler,
                            train_dataloader, eval_dataloader,
                            evaluator,
                            global_step,
                            model_type="tatitok",
                            clip_tokenizer=clip_tokenizer,
                            clip_encoder=clip_encoder,)
        # Stop training if max steps is reached.
        if global_step >= config.training.max_train_steps:
            accelerator.print(
                f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
            )
            break

    accelerator.wait_for_everyone()
    # Save checkpoint at the end of training.
    save_checkpoint(model, output_dir, accelerator, global_step, logger=logger)
    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if config.training.use_ema:
            ema_model.copy_to(model.parameters())
        model.save_pretrained_weight(output_dir)
    accelerator.end_training()


if __name__ == "__main__":
    main()
