# Split layers (fork-and-merge)

A `split.{op}({branch_1}, {branch_2})` is a **fork-and-merge** node:
both branches see the same input, run independently, and their outputs
are combined channel-wise by `op`. It is the recursive-tree successor to
the old [`skip`](skip.md) pseudo-layer — every residual skip is a Split
with a `pass` branch, and `split.mul` additionally expresses gating
(GeGLU / SwiGLU / self-gating) that skips never could.

This doc is the design reference. See [`layers.md`](layers.md) for the
one-paragraph summary and [`architecture.md`](architecture.md) for how
`Model2Def` (the recursive layer-DAG that hosts Splits) fits the
pipeline.

**JAX only.** `SplitDef` has a `build_jax` but no PyTorch `build_module`;
`Model2` runs on the JAX backend.

## Spec and semantics

A Split occupies a position in the layer chain like any other layer:

    bytes|dense.32.gelu-split.add(dense.32.gelu-dense.32.gelu, pass)-dense.64.tanh

Each branch is a layer-list (the same grammar as the top chain) or the
keyword `pass` for an empty (identity) branch. The parser
([`spec_parser.parse_model2`](../texmo/spec_parser.py)) recurses into
every branch, so Splits nest arbitrarily.

Both branches consume the same input vector. Their outputs are merged
channel-wise:

| op | merge | output size |
|----|-------|-------------|
| `add` | sum the overlapping channels; the longer vector's tail passes through | `max(d₁, d₂)` |
| `cat` | concatenate | `d₁ + d₂` |
| `mul` | multiply the overlapping channels; longer tail passes through | `max(d₁, d₂)` |

The `add` / `cat` merge functions are byte-for-byte the legacy
`_merge_add` / `_merge_cat` from `model_jax.py`, so a skip-form conf
translated to split-form preserves its loss exactly.

### Sequence-length alignment

Branches may consume different numbers of leading positions (e.g. one
branch has a `suffix.2`, the other doesn't). Each branch's `forward`
returns `T − consumed` positions; the merge trims the leading positions
of the less-consuming branches to align with the most-consuming one. The
Split's own `length` is `max(branch.length)`, so the enclosing
`total_padding` accounting is unchanged.

## Two families

Splits split (sorry) into two independent neighbor families that never
cross-mutate into each other:

- **Residual** — `add` / `cat`, canonical form *transform-first,
  `pass`-second*. This is the skip analog: `split.add(F(x), pass) = x +
  F(x)`. `add ↔ cat` swap freely.
- **Gate** — `mul`, canonical form *value-first, gate-second*, the gate
  branch is never `pass` and is size-matched to the value branch.
  Expresses `value(x) ⊙ gate(x)` — GeGLU (`mul(dense.X.gelu, dense.X)`),
  self-gating (`mul(pass, dense.X.σ)`), bilinear (`mul(dense.X, dense.X)`).

## Canonical form and validity

`SplitDef.is_valid` (plus the per-branch `LayerSeqDef.is_valid`) enforces:

1. **Exactly 2 branches.** The 2-way limit lives in `is_valid`, not the
   data model (`branches` is a list), so multi-way is a one-line
   relaxation later.
2. **Each branch is valid**, carries the Split's `input_size` (all
   branches consume the same input), and **doesn't end in a
   suffix-like** (`length > 1`) layer. The last rule mirrors the old
   "merge point can't be right after a suffix": the merge consumes each
   branch's final position. Branch carve-outs (see
   `LayerSeqDef.is_valid`): a branch may **end in a bare `dense.X`**
   (the merge doesn't absorb the projection — the GeGLU linear path),
   and may **start with a normalization** (the pre-norm residual
   pattern, `split.add(rmsnorm-…, pass)`). More generally a bare dense
   is legal exactly when its consumer doesn't open with a full linear
   projection (`projects_input`) — so `dense.X-conv.4-rglru.B`
   (Griffin's linear front-end) is valid anywhere.
3. **Pass position is canonical.** `add` / `cat`: branch 0 (the
   transform) is non-empty, branch 1 is `pass`. `mul`: branch 1 (the
   gate) is non-empty (never `pass`), branch 0 is the value. This also
   rules out the degenerate all-`pass` split.
4. **`mul` branches share an output size** — the gate is elementwise.
5. **Bilinear dedup.** A bare-dense *value* path (branch 0) is allowed
   only when the gate is also bare dense (the symmetric bilinear case);
   otherwise the activated path goes first. This collapses the
   commutative GeGLU pair `mul(dense.X.gelu, dense.X)` /
   `mul(dense.X, dense.X.gelu)` to one canonical spelling.

## Skip → split translation

The parser rewrites legacy `skip.D.op` syntax to split form, recursively
at every nesting level (`spec_parser._translate_skips`):

    skip.D.op-A₁-A₂-…-A_D-Y   ⇒   split.op(A₁-A₂-…-A_D, pass)-Y

- **Nested (laminar) skips** — one span fully inside another — become
  **nested splits**; the main branch is translated recursively.
- **Crossing spans** (partial overlap, neither contains the other) can't
  be expressed as a tree → `ValueError`.
- **Overshoot** (span runs past the chain end) → `ValueError`.

`parse_model2` rebuilds the canonical spec from the parsed tree, so a
legacy skip spec and its split form round-trip identically. The result
DB has been fully migrated to split-form (see the split-layer migration
notes), but skip syntax in a template or on the CLI still parses.

## Neighbor mutations

Neighbor generation is recursive over the tree
([`model2.py`](../texmo/model2.py)): every single-layer mutation is
emitted as a spec string and re-parsed through `parse_model2`, so size
threading and `is_valid` filtering live in one place. The generic
sequence mutations (mutate/append/remove a layer, insert/remove
`suffix.2` or `norm`, prepend/remove a dense lead-in) apply inside any
branch *and* the top chain. The Split-specific moves:

**Residual family** (`_split_variants`, `_seq_variants` sec. 10, 12, 13):
- **`add ↔ cat`** op-swap, branches unchanged.
- **Mutate a non-`pass` branch's** layers (recurse).
- **Wrap** a 1- or 2-layer span into `split.op(span, pass)`.
- **Unwrap** a split to its main branch (drop the residual).
- **Grow / shrink** the main span by moving its right boundary across
  the adjacent layer — the `skip` distance ±1 analog. A step that would
  end the branch on a suffix is filtered by `is_valid` rather than
  bumped over (the suffix skip-over is deferred), so a residual spanning
  a suffix needs a longer walk.

**Gate family** (`_split_variants` mul branch, `_seq_variants` sec. 11, 14):
- **Wrap a single layer** in `split.mul(layer, dense.size)`.
- **Toggle the gate activation** (`{none, gelu, relu, tanh}`) and the
  **value↔pass self-gate** form.
- **Append a trailing self-gate** `split.mul(pass, dense.size[.act])` on
  the running activation — reachable even from the empty chain (its
  inverse is the unwrap above).
- **Mutate the value (main) path** with the gate held fixed. A mutation
  that changes the value's size yields an unequal-size `mul` that
  `is_valid` rejects, so the coupled gate-resize isn't offered in one
  hop — it's reachable in two (the invalid intermediate's own neighbors
  resize the gate), which a depth ≥ 2 walk finds for free.

## Counts

- `num_weights` = Σ over branches (Split itself is weightless).
- `num_mults` = Σ over branches + a small merge term `size · (n − 1)`.
- `num_layers` = `1` (the Split, the structural analog of the old
  `SkipDef` pseudo-layer) + Σ branch `num_layers`. This makes a
  skip-translated spec match the legacy `ModelDef.num_layers` exactly —
  e.g. `dense-skip.1.add-dense-dense` (4) → `dense-split.add(dense,
  pass)-dense`, where the split contributes `1 + 1 + 0 = 2`, total 4.

## Runtime (`SplitJax`)

Per-branch weights and state are **lists** (one entry per branch); JAX
walks them as pytrees, so grads / jit / vmap pass through with no
per-layer plumbing.

- `step` — each branch steps on the same `x` and updates its own state;
  outputs merge element-wise (one vector per branch, no temporal
  alignment needed).
- `forward` — each branch runs on the full `inputs`; the less-consuming
  branches' leading positions are trimmed before the channel-axis merge.

`branch_consumed[i] = branch.length − 1` is computed by `SplitDef` and
passed in, so the runtime never inspects heterogeneous branch internals.

## Implementation shape

- **`SplitDef`** (`layers/split.py`) — `op`, `branches: list[LayerSeqDef]`,
  derived `size` / `length`, `__str__` (renders an empty branch as
  `pass`), `num_weights` / `num_mults` / `num_layers`, `is_valid`,
  `build_jax`. No `build_module` (JAX only).
- **`SplitJax`** (`layers/split.py`) — the runtime above.
- **`LayerSeqDef`** (`layers/seq.py`) — a branch or the top chain;
  `is_valid(allow_terminal_bare_dense=…)`, recursive `neighbors`.
- **`Model2Def`** (`model2.py`) — the model descriptor that hosts the
  top `LayerSeqDef`; `neighbors()` generates and `is_valid`-filters.
- **`parse_model2` / `_translate_skips`** (`spec_parser.py`) — the single
  construction entry point and the skip-to-split rewrite.

## Deferred / open

- **Multi-way** (N > 2) branches — one `is_valid` relaxation.
- **Coupled gate-resize** in a single neighbor hop (reachable in two
  today).
- **Suffix-spanning residual** via a distance bump (skip's old skip-over
  behavior).
- **Non-power-of-2 self-gate from the empty chain** — the size-matched
  gate dense is rejected by the dense power-of-2 rule for inputs like
  `bits.2.oh+bp` (width 6) / `bits.4.oh+bp` (17). Add a carve-out only if
  it proves valuable.
- **v2 loss predictor with real branching.** The current predictor
  flattens the tree skip-style: `split.add` / `split.cat` reuse the
  `skip_add` / `cat_dist` feature slots (so all legacy skip data
  transfers), and `split.mul` gets its own slot. A faithful tree-shaped
  predictor would use real "split" features.
