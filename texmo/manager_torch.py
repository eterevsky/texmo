import logging
import math
from time import perf_counter
from typing import Optional

import numpy as np
import torch

from .common import INF, ttoa3, is_power2_int
from .configuration import Configuration
from .dataset import DataSet, DataSetWrapper
from .model_torch import ModelDef, Model, build_model_def
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


class ManagerTorch(object):
    def __init__(
        self,
        conf: Configuration,
        system: str,
        dataset: DataSet,
        device: str = 'auto',
        test_sample_len: int = 1024,
        test_batch: int = 1024,
    ):
        assert isinstance(system, str)
        assert isinstance(conf, Configuration)
        assert isinstance(dataset, (DataSet, DataSetWrapper))

        self.conf = conf
        self._system = system
        self.dataset = dataset
        self.dtype = conf.precision.dtype
        self.device = _resolve_device(device)
        self.test_sample_len = test_sample_len
        self.test_batch = test_batch

        self.model_def = build_model_def(str(conf.model), precision=conf.precision)
        tokenizer = get_tokenizer(self.model_def.input.tokens_name)
        self.bytes_per_token = tokenizer.tokenset.avg_bytes_per_token
        self.model: Optional[Model] = None
        self.optimizer = None
        self.run: Optional[Run] = None

    @property
    def step(self):
        return self.run.steps

    @property
    def loss(self) -> float:
        return self.run.loss

    def init(self, quiet=False):
        logging.info(f'{self.conf}')

        self.model = self.model_def.build_model()
        self.model.to(self.device)

        if not quiet:
            logging.info('Creating optimizer')
        self._build_optimizer()
        self.run = Run(loss_trend=LossTrend(), system=self._system)

    def _build_optimizer(self):
        lr = self.conf.lr
        if self.conf.decay != 1:
            # We'll set up a LambdaLR scheduler
            decay = self.conf.decay
            steps = self.conf.steps
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=0.01,
            )
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: decay ** (step / steps),
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=0.01,
            )
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

    def train(
        self,
        steps: Optional[int],
        time_limit: Optional[float],
        temp_steps=None,
        temp_dir=None,
        quiet=False,
        soft_tl: Optional[float] = None,
    ):
        last_report = 0

        if steps is None and time_limit is None:
            steps = self.conf.steps

        if steps is None:
            steps = INF

        t = '' if time_limit is None else f' {time_limit} s'
        s = '' if steps > 1e10 else f' {steps} steps'
        logging.info(f'Training for{t}{s}')

        deadline = INF
        soft_deadline = INF
        start_time = None

        while self.step < steps:
            if perf_counter() > deadline:
                logging.info(
                    f'Stopped at step {self.step} due to hard time limit {ttoa3(time_limit)}')
                break
            if (
                self.step & (self.step - 1) == 0  # Step is power of 2
                and perf_counter() > soft_deadline
            ):
                logging.info(
                    f'Stopped at step {self.step}/{steps} due to soft time limit {ttoa3(soft_tl)}')
                break

            batch = self._get_batch()
            loss = self.train_step(batch)

            step_end = perf_counter()

            if math.isnan(loss) or math.isinf(loss):
                logging.info(f'Stopping training, loss: {loss}')
                start_time = None
                break

            if not quiet and (
                self.step < 10
                or (self.step % 10 == 0 and step_end - last_report > 3)
                or step_end - last_report > 10
            ):
                last_report = step_end
                logging.info(f'{self.step}  {loss:.4f} b/B')

            if start_time is None:
                start_time = step_end
                deadline = start_time + time_limit if time_limit else INF
                soft_deadline = start_time + soft_tl if soft_tl else INF

        total_time = (
            None if start_time is None else perf_counter() - start_time
        )

        return total_time, self.conf.replace(steps=self.step)

    @torch.no_grad()
    def eval(self) -> float:
        """Evaluate the model on a random test batch.

        Uses test_sample_len and test_batch, which can differ from the
        training length and batch size.
        """
        self.model.eval()
        data = self.dataset.sample_tokens(
            ntokens=self.test_sample_len,
            batch=self.test_batch,
            tokenset_name=self.model_def.input.tokens_name,
        )
        batch = torch.from_numpy(data).long().to(self.device)
        loss = self.model.loss_batch(batch)
        return loss.item() / self.bytes_per_token

    def train_and_eval(
        self,
        steps: Optional[int],
        time_limit: Optional[float],
        temp_steps,
        temp_dir,
        output_dir,
        log,
        quiet: bool = False,
        soft_tl: Optional[float] = None,
    ) -> tuple[Run, Configuration]:
        try:
            train_time, final_conf = self.train(
                steps,
                time_limit,
                temp_steps,
                temp_dir,
                quiet=quiet,
                soft_tl=soft_tl,
            )

            if train_time is None:
                eval_loss = INF
            else:
                eval_loss = self.eval()
                if math.isnan(eval_loss):
                    eval_loss = INF
        except Exception:
            logging.warning('Training stopped early.')
            eval_loss = INF
            train_time = None
            final_conf = self.conf.replace(steps=self.step)

        self.run.finalize(eval_loss, train_time)
        if final_conf != self.conf:
            steps_log = f'  after {final_conf.steps} steps'
        else:
            steps_log = ''
        logging.info(
            f'loss {eval_loss:.4f} b/byte  T = {ttoa3(train_time)}{steps_log}')

        return (self.run, final_conf)

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

    def name(self) -> str:
        return str(self.model_def)
