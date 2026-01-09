import hydra
from omegaconf import OmegaConf
from pathlib import Path
from peft import get_peft_model

import torch
import psutil
import os

from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer
from f5_tts.model import CFM

from src.voicehs.train.lora import DiTLora, TrainerLora, lora_config

project_root = Path(__file__).parent.parent.parent.parent  # project root
weights_dir = project_root / "ckpts"
weights_dir.mkdir(parents=True, exist_ok=True)

def print_mem(tag=""):
    print(f"[MEM {tag}] RAM: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.1f} MB | "
          f"GPU: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

def check_lora_params(model):
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"[LoRA PARAMS] Trainable params count: {trainable_count}/{total_count} "
          f"({trainable_count / total_count * 100:.6f}%)")  # usually <<1% for LoRA

@hydra.main(version_base="1.3", config_path= str(project_root / "src" / "voicehs" / "configs"), config_name=None)
def main(model_cfg):
    model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    tokenizer = model_cfg.model.tokenizer
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
    exp_name = f"{model_cfg.model.name}_{mel_spec_type}_{model_cfg.model.tokenizer}_{model_cfg.datasets.name}"
    wandb_resume_id = None

    # Tokenizer
    if model_cfg.model.tokenizer != "custom":
        tokenizer_path = model_cfg.datasets.name
    else:
        tokenizer_path = model_cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    # LoRA-injected Model
    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=model_cfg.model.mel_spec.n_mel_channels),
        mel_spec_kwargs=model_cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )
    transformer = model.transformer
    transformer = DiTLora(transformer)
    transformer = get_peft_model(transformer, lora_config) # type: ignore
    model.transformer = transformer

    print_mem("after model init")
    check_lora_params(model)

    checkpoint_dir = weights_dir / "ckpts"  # this mirrors your structure weights/ckpts/
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Trainer
    trainer = TrainerLora(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=str(checkpoint_dir),
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project="VoiceHS",
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True)
    )

    print("Trainer initialized")
    print("Trainer model type:", type(trainer.model))
    print("Optimizer type:", type(trainer.optimizer))

    # dataset
    train_dataset = load_dataset(model_cfg.datasets.name, tokenizer, mel_spec_kwargs=model_cfg.model.mel_spec)

    print_mem("before training")
    trainer.train(train_dataset, num_workers=model_cfg.datasets.num_workers, resumable_with_seed=42)
    print_mem("after training")

if __name__ == "__main__":
    main()
