from typing import Any, Optional

from .common import power2_neighbors

# Window used when a type swap lands on attn from a layer that has no
# window of its own (msr <-> attn). Modest starting point; the window
# then mutates freely (x2 / /2) within attn.
_ATTN_SWAP_WINDOW = 16
from .layer_jax import LayerJax

LayerState = Any


class LayerDef(object):
    """Descriptor for a layer. Lightweight, no weights."""
    name = "OVERRIDE"

    def __init__(self, input_size: int):
        self.input_size: int = input_size
        self.size: Optional[int] = None
        # The number of the output steps from the previous layers on which
        # this layer depends. Should be 1 for dense and various recursive
        # layers, and length of the suffix for suffix and attn layers.
        self.length: int = 1

    def __eq__(self, other):
        return str(self) == str(other)

    @property
    def num_weights(self) -> int:
        raise NotImplementedError

    @property
    def num_mults(self) -> int:
        """Multiplications per (batch=1, length=1) token through one
        forward pass. Used as a workload proxy — `num_weights` is a
        good first approximation (each non-bias weight does one mult
        per token; the bias adds count as 'honorary' mults). Layers
        that reuse the same weight matrix multiple times per token
        (latent, lrnn) override this."""
        return self.num_weights

    @property
    def projects_input(self) -> bool:
        """True when the layer's first operation on its input is a full
        linear projection (W @ x) — so a preceding bare (activation-less)
        dense would collapse into that projection and is degenerate.
        Used by `LayerSeqDef.is_valid` to decide where a bare dense is
        allowed. Default True (the conservative choice for new layers);
        overridden False by layers that consume their input elementwise
        or per-channel (conv, suffix, norm, rmsnorm, rglru) and by
        Split, which projects only if every branch head does."""
        return True

    @property
    def num_layers(self) -> int:
        """Count of layers contributed by this LayerDef. Leaf layers
        report 1; structural layers (LayerSeqDef holds N children;
        SplitDef holds itself plus its branches) override to recurse
        through their contents so the total matches the flat count on
        the spec (a skip pseudo-layer and each spanned layer counted
        1 apiece in the retired flat-list representation)."""
        return 1

    def is_valid(self) -> bool:
        raise NotImplementedError

    def neighbors(self):
        """Yield spec strings for single-mutation neighbors.

        Size/length 2x changes and swaps between related layer types:
            dense <-> rnn (with activation preserved)
            rnn <-> {gru, mgru, mingru} (activation dropped/picked)
            gru, mgru, mingru, lstm are mutual neighbors (except
            lstm <-> mingru and lstm <-> rnn, which are not)
            dense.X.tanh <-> latent.X.2
            rnn.X.tanh <-> lrnn.X.2
            latent.X.Y <-> lrnn.X.Y (and size/reps 2x mutations)
        """
        if self.name == "suffix":
            for l in power2_neighbors(self.length):
                yield f"suffix.{l}"
            # Type swap: suffix.L <-> conv.L (same time-axis width;
            # suffix downsamples by L, conv preserves length).
            yield f"conv.{self.length}"
            # Type swap: suffix.L <-> single-head attn over the same
            # span -- the learned-soft wrap of the last L positions vs
            # the hard concat. Size = the suffix's input width.
            yield f"attn.{self.input_size}.1.{self.length}"
            return

        if self.name == "conv":
            for l in power2_neighbors(self.kernel):
                yield f"conv.{l}"
            # Type swap: conv.L <-> suffix.L (mirror of the suffix
            # case above).
            yield f"suffix.{self.kernel}"
            # Type swap: conv.L <-> single-head attn over the same
            # span -- the two length-preserving mixers, a learned
            # static FIR vs a content-dependent lookup. Size = the
            # conv's input width (depthwise preserves it), window =
            # the kernel. Like the suffix edge, input widths that make
            # the attn spec invalid are left to downstream is_valid.
            yield f"attn.{self.input_size}.1.{self.kernel}"
            return

        if self.name in ("norm", "rmsnorm"):
            # Swap between the parameter-free L2 norm and the learned-
            # scale RMS norm; both are size-preserving with no other
            # metaparameter.
            yield "rmsnorm" if self.name == "norm" else "norm"
            return

        if self.name in ("latent", "lrnn", "lmgu"):
            # Size 2x (keep > 1)
            for s in power2_neighbors(self.size):
                if s > 1:
                    yield f"{self.name}.{s}.{self.reps}"
            # Reps 2x (keep >= 2)
            for r in power2_neighbors(self.reps):
                if r >= 2:
                    yield f"{self.name}.{self.size}.{r}"
            # Swaps among the depth-recurrent family at same (S, R).
            if self.name == "latent":
                yield f"lrnn.{self.size}.{self.reps}"
            elif self.name == "lrnn":
                yield f"latent.{self.size}.{self.reps}"
                yield f"lmgu.{self.size}.{self.reps}"
            elif self.name == "lmgu":
                yield f"lrnn.{self.size}.{self.reps}"
            # Collapse to non-iterating cousin at reps == 2.
            if self.reps == 2:
                if self.name == "latent":
                    yield f"dense.{self.size}.tanh"
                elif self.name == "lrnn":
                    yield f"rnn.{self.size}.tanh"
                elif self.name == "lmgu":
                    yield f"mgru.{self.size}"
            return

        if self.name == "msr":
            # The `.nope` qualifier (rotation off) rides along every
            # metaparameter mutation; only its own toggle edge below
            # changes it. Bare spelling means RoPE.
            nope = ".nope" if self.nope else ""
            # 2x dim. The floor is per-flag: RoPE needs at least one
            # interleaved pair (dim >= 2), while nope has no pairs and
            # bottoms out at a 1x1 per-head state.
            min_dim = 1 if self.nope else 2
            for d in power2_neighbors(self.dim):
                if d >= min_dim:
                    yield f"msr.{d}.{self.heads}{nope}"
            # 2x heads (must stay >= 1).
            for h in power2_neighbors(self.heads):
                yield f"msr.{self.dim}.{h}{nope}"
            # Weightless position-encoding toggle. Dropping the
            # rotation only relaxes bounds, so rope -> nope always
            # applies; the way back is skipped when the rope twin would
            # be invalid (dim 1 has no rotation pair), yielding nothing
            # rather than an invalid spec. Sub-rope dims stay connected
            # to the rest of nope-space through the mutations above.
            if not self.nope:
                yield f"msr.{self.dim}.{self.heads}.nope"
            elif self.dim >= 2:
                yield f"msr.{self.dim}.{self.heads}"
            # Single-head msr swaps with mgru of the same size. RoPE
            # form only: mgru has no position encoding for the flag to
            # carry over to, and the reverse edge (in the recurrent
            # branch) names the bare spelling, so restricting to it
            # keeps the pair an exact image. The nope form reaches
            # mgru by toggling first.
            if self.heads == 1 and not self.nope:
                yield f"mgru.{self.dim}"
            # Attention-family swap: linear attention <-> softmax
            # attention at the same total width and head count. msr's
            # first field is the *per-head* dim, attn's is the total
            # size, so the swap goes through self.size (= H*D), not
            # self.dim. The swapped layer's head_dim is this dim, and
            # attn needs head_dim >= 4 for a rotary pair -- below that
            # there is no equal-shape twin, so yield nothing. attn also
            # needs a window msr has no analogue for; start modest and
            # let it mutate.
            # The swap is flag-preserving: rope <-> rope, nope <-> nope.
            # The twin's head_dim is this dim, so the guard follows the
            # target's per-flag floor -- 4 for rope, 1 for nope -- and
            # each flag class stays an exact image of itself.
            if self.dim >= (1 if self.nope else 4):
                yield (f"attn.{self.size}.{self.heads}."
                       f"{_ATTN_SWAP_WINDOW}{nope}")
            return

        if self.name == "attn":
            # 2x mutations on all three metaparameters; is_valid
            # filters the boundary cases (head_dim >= 4, window >= 2).
            # `.nope` rides along every metaparameter mutation (mirror
            # of the msr case above).
            nope = ".nope" if self.nope else ""
            for s in power2_neighbors(self.size):
                yield f"attn.{s}.{self.heads}.{self.window}{nope}"
            for h in power2_neighbors(self.heads):
                yield f"attn.{self.size}.{h}.{self.window}{nope}"
            for w in power2_neighbors(self.window):
                yield f"attn.{self.size}.{self.heads}.{w}{nope}"
            # Weightless position-encoding toggle (mirror of msr's):
            # rope -> nope always applies since it only relaxes bounds,
            # while nope -> rope is skipped below the rotary floor
            # (head_dim < 4) rather than naming an invalid spec.
            if not self.nope:
                yield f"attn.{self.size}.{self.heads}.{self.window}.nope"
            elif self.head_dim >= 4:
                yield f"attn.{self.size}.{self.heads}.{self.window}"
            # Attention-family swap (mirror of the msr case above):
            # attn's total size splits into msr's per-head dim, so the
            # widths match (H * head_dim == size). head_dim is 0 when
            # heads doesn't divide size, so the >= 4 guard covers
            # divisibility too and keeps this edge the exact image of
            # the forward swap.
            # Flag-preserving, like the forward direction. The floor is
            # the target msr's per-flag one (2 for rope, 1 for nope);
            # head_dim is 0 when heads doesn't divide size, so either
            # floor also covers divisibility.
            if self.head_dim >= (1 if self.nope else 4):
                yield f"msr.{self.head_dim}.{self.heads}{nope}"
            # Mirror of the suffix and conv swaps: single-head attn over
            # window W at the input width <-> suffix.W (hard concat of
            # the same span) and <-> conv.W (static FIR over it).
            # Guarded to the exact image of the forward swaps so the
            # relations stay symmetric. Both floors are 2 (attn's
            # window, conv's kernel), so W transfers without a guard.
            # RoPE form only: the forward bridges (in the suffix and
            # conv branches) name the bare spelling, so the nope form
            # has no bridge and reaches one by toggling first.
            if (self.heads == 1 and self.size == self.input_size
                    and not self.nope):
                yield f"suffix.{self.window}"
                yield f"conv.{self.window}"
            return

        if self.name == "rglru":
            # The only metaparameter is the block count (power of 2);
            # is_valid filters candidates that don't divide the width.
            for b in power2_neighbors(self.blocks):
                yield f"rglru.{b}"
            # Size-preserving swap with the minimal input-gated cells, at
            # the canonical single-block form: rglru.1 <-> mgru.X /
            # mingru.X (which preserve size when X == input). rglru.size
            # == input_size by construction, so the targets are size-safe.
            if self.blocks == 1:
                yield f"mgru.{self.size}"
                yield f"mingru.{self.size}"
            return

        # Size mutations (keep same layer type and activation)
        _LSTM_FAMILY = ("lstm", "matlstm", "slstm", "mullstm")
        if self.name in ("dense", "rnn"):
            # A bare dense (linear/gate path inside a split branch) has
            # no activation -- render without the suffix, else the
            # f-string emits a literal ".None" that fails to parse.
            act = f".{self._activation}" if self._activation else ""
            for s in power2_neighbors(self.size):
                yield f"{self.name}.{s}{act}"
            # Activation mutations at the same size: every activation
            # other than the current one, so the whitelist forms a
            # complete graph and each is one mutation from each. For
            # dense the bare (linear) form is part of the cycle -- it
            # is the tied-codec adapter and the gate/linear path,
            # filtered by validity everywhere else. rnn has no bare
            # form. A relu layer (retired, search-invalid) proposes the
            # survivors, so existing relu lineages migrate instead of
            # dying. Kept in step with the `is_valid` whitelists in
            # layers/dense.py and layers/rnn.py -- importing them here
            # would be a genuine cycle (both import this module).
            if self.name == "dense":
                acts = (None, "tanh", "gelu", "silu")
            else:
                acts = ("tanh", "gelu", "silu")
            for a in acts:
                if a != self._activation:
                    suffix = f".{a}" if a else ""
                    yield f"{self.name}.{self.size}{suffix}"
        elif self.name in ("gru", "mgru", "mingru", "lstm", "slstm", "mullstm"):
            for s in power2_neighbors(self.size):
                yield f"{self.name}.{s}"
        elif self.name == "matlstm":
            # matlstm needs size >= 2 (matrix cell isn't meaningful at D=1).
            for s in power2_neighbors(self.size):
                if s >= 2:
                    yield f"{self.name}.{s}"

        # Type swaps
        _RECURRENT = ("gru", "mgru", "mingru")
        if self.name == "dense":
            # Cross-type swaps need an activation to carry over; a bare
            # dense (linear path) stays a dense.
            if self._activation:
                yield f"rnn.{self.size}.{self._activation}"
                if self._activation == "tanh" and self.size > 1:
                    yield f"latent.{self.size}.2"
        elif self.name == "rnn":
            # A bare (activation-less) rnn is search-invalid but still
            # parses, like a retired relu form, so its twin has to be
            # spelled `dense.X` -- interpolating the activation
            # unguarded emitted a literal "dense.X.None" that failed to
            # re-parse. The reverse edge (dense branch above) stays
            # guarded instead of mirrored: a bare dense has no rnn
            # twin, because the bare rnn it would name is invalid.
            act = f".{self._activation}" if self._activation else ""
            yield f"dense.{self.size}{act}"
            for other in _RECURRENT:
                yield f"{other}.{self.size}"
            if self._activation == "tanh" and self.size > 1:
                yield f"lrnn.{self.size}.2"
        elif self.name in _RECURRENT:
            # Every search-valid rnn activation, matching the cycle
            # above -- wherever one is offered, all are.
            for act in ("gelu", "tanh", "silu"):
                yield f"rnn.{self.size}.{act}"
            for other in _RECURRENT:
                if other != self.name:
                    yield f"{other}.{self.size}"
            # lstm <-> gru, mgru (but not mingru)
            if self.name in ("gru", "mgru"):
                yield f"lstm.{self.size}"
            # mgru <-> msr.X.1 (single-head retention has the same
            # vector state width as mgru.X). msr needs dim >= 2 for
            # at least one RoPE pair, so skip when mgru.1.
            if self.name == "mgru" and self.size >= 2:
                yield f"msr.{self.size}.1"
            # mgru <-> lmgu.X.2 (the depth-recurrent variant collapses
            # to mgru at reps=2). lmgu needs size > 1.
            if self.name == "mgru" and self.size > 1:
                yield f"lmgu.{self.size}.2"
            # mgru.X / mingru.X <-> rglru.1: the RG-LRU is size-preserving,
            # so the swap is only size-safe when this cell preserves size
            # (size == input). Lands on the canonical single-block form.
            if (self.name in ("mgru", "mingru")
                    and self.size == self.input_size):
                yield "rglru.1"
        elif self.name in _LSTM_FAMILY:
            # All-to-all within the LSTM family (lstm/matlstm/slstm/
            # mullstm). matlstm has size >= 2 constraint; skip the
            # swap target when current size is 1.
            for other in _LSTM_FAMILY:
                if other == self.name:
                    continue
                if other == "matlstm" and self.size < 2:
                    continue
                yield f"{other}.{self.size}"
            # lstm also keeps its existing swaps with gru/mgru.
            if self.name == "lstm":
                yield f"gru.{self.size}"
                yield f"mgru.{self.size}"

    def build_jax(self, dtype) -> LayerJax:
        """Create a JAX layer implementation."""
        raise NotImplementedError
