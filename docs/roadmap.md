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

* **Incremental model training** — port the `add_layers` feature from the
  old manager (freezes existing weights, trains only the new ones).
  Integrate into search as a strategy: instead of training a deep model
  from scratch, build it up layer by layer.
