from peft import LoraConfig, TaskType
import torch

from f5_tts.model.trainer import Trainer
from ema_pytorch import EMA


class DiTLora(torch.nn.Module):
    def __init__(self, transformer: torch.nn.Module):
        super().__init__()
        self.model = transformer

    def forward(self, *args, **kwargs):
        kwargs.pop("input_ids", None)
        kwargs.pop("attention_mask", None)
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("output_attentions", None)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict", None)
        return self.model(*args, **kwargs)

    # PEFT hooks
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return kwargs

    def _prepare_encoder_decoder_kwargs_for_generation(self, *args, **kwargs):
        return {}

    # delegate attribute access to wrapped transformer
    def __getattr__(self, name):
        if name == "model":  # avoid recursion
            return super().__getattr__(name)
        return getattr(self.model, name)


class TrainerLora(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ensure required attributes exist
        self.learning_rate = kwargs.get("learning_rate", getattr(self, "learning_rate", 1e-3))
        self.max_grad_norm = kwargs.get("max_grad_norm", getattr(self, "max_grad_norm", 1.0))
        self.accelerator = getattr(self, "accelerator", None)

        lora_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(lora_params, lr=self.learning_rate)
        if getattr(self, "is_main", True):
            self.ema_model = EMA(self.model, beta=0.999)
            self.ema_model.to(self.accelerator.device if self.accelerator is not None else None)

        # Re-wrap for distributed training
        if self.accelerator is not None:
            self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)


def peftify(transformer: torch.nn.Module) -> torch.nn.Module:
    """
    Patches a transformer model so it works with PEFT LoRA without wrapping.
    Adds the minimal hooks PEFT expects for generation.
    """
    # only patch if not already present
    if not hasattr(transformer, "prepare_inputs_for_generation"):
        transformer.prepare_inputs_for_generation = lambda *args, **kwargs: kwargs # type: ignore
    if not hasattr(transformer, "_prepare_encoder_decoder_kwargs_for_generation"):
        transformer._prepare_encoder_decoder_kwargs_for_generation = lambda *args, **kwargs: {} # type: ignore
    return transformer


target_modules = []
for i in range(22):  # you have 0..21 blocks
    target_modules += [
        f"model.transformer_blocks.{i}.attn.to_q",
        f"model.transformer_blocks.{i}.attn.to_k",
        f"model.transformer_blocks.{i}.attn.to_v",
        f"model.transformer_blocks.{i}.attn.to_out.0",
        f"model.transformer_blocks.{i}.ff.ff.2",
    ]

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,
)

def _strip_input_ids_forward(self, *args, **kwargs):
    kwargs.pop("input_ids", None)
    return self.model.forward(*args, **kwargs)
