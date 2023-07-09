import math
import numpy as np

# We aren't differentiating losses above 10 bits per byte.
MIN_LOSS = 0.1
MAX_LOSS = 10
MAX_LOG_LOSS = math.log2(MAX_LOSS)


def encode_loss(loss: np.ndarray):
    if loss is None:
        return None
    loss = np.minimum(loss, 2**16)
    loss = np.log2(loss)
    return loss


def decode_loss(loss):
    if loss is None:
        return None
    loss = np.minimum(loss, MAX_LOG_LOSS)
    return np.exp2(loss)


def prediction_score(true_losses: np.ndarray, predicted_losses: np.ndarray):
    n = len(true_losses)
    assert len(predicted_losses) == n

    all_losses = np.concatenate((true_losses, predicted_losses))
    all_losses = np.minimum(all_losses, MAX_LOSS)
    all_losses = np.maximum(all_losses, MIN_LOSS)
    all_losses = np.log2(all_losses)

    return np.average(np.abs(all_losses[:n] - all_losses[n:]))

