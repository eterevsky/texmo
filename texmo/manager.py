import logging
import math
from time import perf_counter
from typing import Optional

from .common import INF, ttoa3
from .configuration import Configuration
from .dataset import DataSet, DataSetWrapper
from .run import Run
from .tokens import get_tokenizer


class Manager:
    """Base class defining the manager interface.

    A manager takes a Configuration, trains a model, evaluates it,
    and produces a Run with the results. The two backends (Torch, JAX)
    implement this interface independently.
    """

    def __init__(
        self,
        conf: Configuration,
        system: str,
        dataset: DataSet,
        test_sample_len: int = 1024,
        test_batch: int = 1024,
        verbose: bool = True,
    ):
        assert isinstance(system, str)
        assert isinstance(conf, Configuration)
        assert isinstance(dataset, (DataSet, DataSetWrapper))

        self.conf = conf
        self.system = system
        self.dataset = dataset
        self.test_sample_len = test_sample_len
        self.test_batch = test_batch
        self.verbose = verbose

        self.model_def = conf.model
        tokenizer = get_tokenizer(self.model_def.input.tokens_name)
        self.bytes_per_token = tokenizer.tokenset.avg_bytes_per_token
        self.run: Optional[Run] = None

    @property
    def step(self):
        return self.run.steps

    @property
    def loss(self) -> float:
        return self.run.loss

    def _get_batch(self):
        """Return a training batch as a tensor appropriate for the backend."""
        raise NotImplementedError

    def train_step(self, batch) -> float:
        """Run one training step. Returns loss in bits per byte."""
        raise NotImplementedError

    def eval(self) -> float:
        """Evaluate the model; returns loss in bits per byte."""
        raise NotImplementedError

    def train(
        self,
        steps: Optional[int],
        time_limit: Optional[float],
    ) -> tuple[Optional[float], Configuration]:
        """Run the training loop.

        Returns:
            (train_time, final_configuration) where train_time is None
            if training diverged.
        """
        last_report = 0

        if steps is None and time_limit is None:
            steps = self.conf.steps

        if steps is None:
            steps = INF

        t = '' if time_limit is None else f' {time_limit} s'
        s = '' if steps > 1e10 else f' {steps} steps'
        logging.info(f'Training for{t}{s}')

        deadline = INF
        start_time = None

        while self.step < steps:
            if perf_counter() > deadline:
                logging.info(
                    f'Stopped at step {self.step} due to time limit {ttoa3(time_limit)}')
                break

            batch = self._get_batch()
            loss = self.train_step(batch)

            step_end = perf_counter()

            if math.isnan(loss) or math.isinf(loss):
                logging.info(f'Stopping training, loss: {loss}')
                start_time = None
                break

            if self.verbose and (
                self.step < 10
                or (self.step % 10 == 0 and step_end - last_report > 3)
                or step_end - last_report > 10
            ):
                last_report = step_end
                logging.info(f'{self.step}  {loss:.4f} b/B')

            if start_time is None:
                start_time = step_end
                deadline = start_time + time_limit if time_limit else INF

        total_time = (
            None if start_time is None else perf_counter() - start_time
        )

        return total_time, self.conf.replace(steps=self.step)

    def continue_prefix(
        self, prefix: str, length: int, temperature: float
    ) -> bytes:
        """Sample text continuation from the model."""
        raise NotImplementedError

    def train_and_eval(
        self,
        steps: Optional[int],
        time_limit: Optional[float],
    ) -> tuple[Run, Configuration]:
        """Train the model and evaluate it.

        Returns:
            (run, final_configuration).
        """
        train_time, final_conf = self.train(steps, time_limit)

        if train_time is None:
            eval_loss = INF
        else:
            eval_loss = self.eval()
            if math.isnan(eval_loss):
                eval_loss = INF

        self.run.finalize(eval_loss, train_time)
        if final_conf != self.conf:
            steps_log = f'  after {final_conf.steps} steps'
        else:
            steps_log = ''
        logging.info(
            f'loss {eval_loss:.4f} b/byte  T = {ttoa3(train_time)}{steps_log}')

        return (self.run, final_conf)


def create_manager(
    backend: str,
    conf: Configuration,
    system: str,
    dataset: DataSet,
    device: Optional[str] = None,
    test_sample_len: int = 1024,
    test_batch: int = 1024,
    verbose: bool = True,
) -> Manager:
    """Create a Manager for the given backend ('torch' or 'jax').

    Uses local imports to avoid circular dependencies — both
    manager_torch and manager_jax import Manager from this module.
    """
    common = dict(
        conf=conf, system=system, dataset=dataset,
        test_sample_len=test_sample_len, test_batch=test_batch,
        verbose=verbose,
    )
    match backend:
        case 'torch':
            from .manager_torch import ManagerTorch
            return ManagerTorch(device=device or 'auto', **common)
        case 'jax':
            from .manager_jax import ManagerJax
            return ManagerJax(**common)
        case _:
            raise ValueError(f"Unknown backend: {backend}")
