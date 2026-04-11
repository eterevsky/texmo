# Roadmap

## Grand goal

Given a parameter budget N and compute budget T, answer:
1. What is the best model architecture?
2. How should it be trained? — i.e. which metaparameters, what LR schedule,
   and potentially whether to use incremental training (adding layers
   gradually) instead of training from scratch.

## Next steps

* **Attention / Transformer layer.** Use PyTorch's built-in
  `nn.MultiheadAttention` or `nn.TransformerEncoderLayer`. Key decisions:
  causal masking, position encoding (RoPE?), head count as a search axis.

* **Add residual connections.** Possibly make a `skip.N.add`/`skip.N.stack`
  layer that takes current activations, skip the next N layers and either
  adds them (if the dimensions match), or stacks them to the later activations.

* **Loss prediction model** taking architecture + metaparameters as input.
  Predict expected loss for a configuration without running it. Could use
  an RNN over the layer sequence, or something like a recurrent random
  forest.

* **Use the loss predictor in search** to skip small mutations whose
  predicted loss is unlikely to improve on the current best. This cuts
  most of the search cost.

* **Incremental model training** — port the `add_layers` feature from the
  old manager (freezes existing weights, trains only the new ones).
  Integrate into search as a strategy: instead of training a deep model
  from scratch, build it up layer by layer.

* **Search strategies beyond random walk** — find a local optimum starting
  from a winning configuration. Improve `train` CLI to automatically
  optimize metaparameters and architecture within given constraints.
