import logging

import torch

from .manager import Manager
from .model import Model
from .predict import LossTrend
from .run import Run
from .tokens import get_tokenizer


def _resolve_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(device_str)


class ManagerTorch(Manager):
    """PyTorch training backend."""

    def __init__(self, device: str = 'auto', **kwargs):
        super().__init__(**kwargs)
        self.dtype = self.conf.precision.dtype
        self.device = _resolve_device(device)

        logging.info(f'{self.conf}')
        self.model: Model = self.model_def.build_model()
        self.model.to(self.device)

        if self.verbose:
            logging.info('Creating optimizer')
        self._build_optimizer()
        self.run = Run(loss_trend=LossTrend(), system=self.system)

    def _build_optimizer(self):
        lr = self.conf.lr
        # Apply weight decay only to weight matrices, not biases.
        decay_params = [p for p in self.model.parameters() if p.dim() >= 2]
        no_decay_params = [p for p in self.model.parameters() if p.dim() < 2]
        param_groups = [
            {'params': decay_params, 'weight_decay': 0.01},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        self.optimizer = torch.optim.AdamW(param_groups, lr=lr)
        if self.conf.cosine:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.conf.steps, eta_min=0.0,
            )
        elif self.conf.decay != 1:
            decay = self.conf.decay
            steps = self.conf.steps
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: decay ** (step / steps),
            )
        else:
            self.scheduler = None

    def _get_batch(self) -> torch.Tensor:
        data = self.dataset.sample_tokens(
            ntokens=self.conf.length,
            batch=self.conf.batch,
            tokenset_name=self.model_def.input.tokens_name,
        )
        return torch.from_numpy(data).long().to(self.device)

    def train_step(self, batch: torch.Tensor) -> float:
        self.model.train()
        loss = self.model.loss_batch(batch)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        loss_val = loss.item() / self.bytes_per_token
        self.run.add_step(loss_val)
        return loss_val

    @torch.no_grad()
    def eval(self) -> float:
        """Evaluate the model on a random test batch."""
        self.model.eval()
        # Evaluate in fp32 for consistent precision across training dtypes.
        self.model.float()
        self.model.input_module.dtype = torch.float32
        data = self.dataset.sample_tokens(
            ntokens=self.test_sample_len,
            batch=self.test_batch,
            tokenset_name=self.model_def.input.tokens_name,
        )
        batch = torch.from_numpy(data).long().to(self.device)
        loss = self.model.loss_batch(batch)
        self.model.to(self.dtype)
        self.model.input_module.dtype = self.dtype
        return loss.item() / self.bytes_per_token

    @torch.no_grad()
    def continue_prefix(
        self, prefix: str, length: int, temperature: float
    ) -> bytes:
        """Sample text continuation from the model."""
        self.model.eval()
        tokenizer = get_tokenizer(self.model_def.input.tokens_name)
        prefix_tokens = tokenizer.tokenize(prefix.encode())

        states, _ = self.model.initial_step()
        for c in prefix_tokens[:-1]:
            states, _ = self.model.step(states, c)

        c = prefix_tokens[-1]
        out = []
        for _ in range(length):
            states, c = self.model.step_sample(states, c, temperature)
            out.append(c)

        return tokenizer.untokenize(list(prefix_tokens) + out)
