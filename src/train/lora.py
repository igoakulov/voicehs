from peft import LoraConfig
import torch

from f5_tts.model.trainer import Trainer
from ema_pytorch import EMA

class CFMLora(torch.nn.Module):
    def __init__(self, cfm_model: torch.nn.Module):
        super().__init__()
        self.model = cfm_model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # PEFT sometimes expects these hooks for generation
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return kwargs

    def _prepare_encoder_decoder_kwargs_for_generation(self, *args, **kwargs):
        return {}


class TrainerLora(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        lora_params = [p for p in self.model.parameters() if p.requires_grad]

        if kwargs.get("bnb_optimizer", False):
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(lora_params, lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.AdamW(lora_params, lr=self.learning_rate)

        # EMA tracks only LoRA parameters
        if self.is_main:
            self.ema_model = EMA(lora_params, beta=0.999)
            self.ema_model.to(self.accelerator.device)

        # Re-wrap optimizer with accelerator for multi-GPU support
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)


target_modules = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.ff.2",
]

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)
