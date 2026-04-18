# Skip layers (residual connections)

A `skip.X.add` or `skip.X.cat` is a pseudo-layer that marks the start
of a residual connection spanning `X` layers. It has no learned
weights. The `Model` / `ModelJax` orchestrator stashes the source
activation and merges it back at the end of the skip.

This doc is the design reference. See [`layers.md`](layers.md) for the
one-paragraph summary.

## Spec and semantics

`skip.X.add` or `skip.X.cat` occupies a position in the spec like any
other layer, e.g.

    bytes|dense.32.gelu-skip.2.add-dense.32.gelu-dense.32.gelu-dense.64.tanh

The skip source is the output of the preceding layer. It is saved and
merged into the activation stream `X` layers later. The skip itself
doesn't transform activations — just passes them through.

`X ≥ 1` (no self-skip) and the merge point must be inside the pipeline
(i.e. `X ≤ remaining_layers_count` after the skip position).

### Merge semantics

Let `d_src` be the source size and `d_cur` the current size at the
merge point.

- **`.add`**: output size is `max(d_src, d_cur)`. The first
  `min(d_src, d_cur)` channels are summed; the remaining channels from
  the longer vector are appended unchanged. This avoids the
  brittleness of strict dim matching — skips survive most size
  mutations.
- **`.cat`**: output size is `d_src + d_cur`.

### Multiple incoming skips at the same merge point

If several skips happen to land on the same position, they are merged
in order of their **start** position (earliest first). This is a
deterministic convention; the order matters when mixing `.add` and
`.cat`.

## Validity rules

A skip is valid iff:

1. `X ≥ 1`.
2. The merge point lands within the pipeline (not past the output
   layer's input — the very last valid merge point is just before the
   output dense).
3. **No two skip pseudo-layers are adjacent** in the spec. This
   prevents an N² explosion of residual starts/ends and matches how
   real architectures use residuals (one per block, not many starting
   at the same position).
4. **Skip source can't be right before a norm.** I.e. the layer
   immediately after the `skip.X.*` pseudo-layer can't be a `norm`.
   Rationale: the source would be an unnormalized activation that gets
   normalized on the main path anyway, and merging an unnormalized
   vector into a normalized main path is semantically off.
5. **Merge point can't be immediately after a suffix.** I.e. the
   last-skipped layer can't be a suffix. Rationale: suffix multiplies
   the activation size by its length, so merging right after it means
   combining a small source with a freshly-inflated target — the
   source contribution would be negligible. Waiting one more layer
   lets a downstream dense bring the dimensionality back down before
   the merge.

There is no restriction on the end point relative to norm — the
search can explore that freely.

## Neighbor mutations

The `LayerDef.neighbors()` for skip yields:

- **Swap `add ↔ cat`**, keeping `X`.
- **Distance `X ± 1`**, as long as the new end stays in the pipeline
  and isn't immediately after a suffix. If `±1` would land right after
  a suffix, use `±2` instead (skip over the post-suffix slot).

At the `ModelDef.neighbors()` level, skip layers participate in:

- **Add a new skip** at any valid position. Preferred distance is
  `X = 1`. The distance is bumped to `X = 2` if `X = 1` would place
  the merge point right after a suffix. (Max bump is `+1` because the
  no-adjacent-suffix invariant means two suffixes can't be in a row.)
- **Remove a skip** (symmetric with add).

**Inserting a norm or suffix in the middle of a skip** must take
that skip into account. Two cases:

- Insertion strictly between a skip's start and (old) end: skip's
  distance is incremented by 1 so the merge still happens at the
  same original position. For a suffix insertion, if that bump
  would leave the merge point right after the new suffix (violating
  rule 5), the insertion isn't a valid neighbor.
- Insertion right *at* the merge point: keep the merge before the
  new layer (distance unchanged). For suffix this is forced by rule
  5. For norm it's a choice: the `norm(main + source)` semantics is
  cleaner (post-norm-like) than `norm(main) + source`.

## Size computation in spec parsing

When `ModelDef` walks the spec, it keeps a map from target position to
the list of pending skips merging there (source size, `.add` / `.cat`,
and start position for ordering). When the walker reaches position P,
it consumes the list for P — earliest start first — and updates the
current size accordingly (`max` for `.add`, `+` for `.cat`). The
resulting `input_size` is what the next real layer sees.

## Weight count

Skip pseudo-layers contribute `0` to `num_weights`.

## Runtime: saving and merging sources

The same save/merge mechanism applies to both `forward` and `step`:

- At a `skip.*` position, record the current activation as a pending
  source for that skip's target position.
- At a target position, merge any pending sources (earliest-start
  first) into the current activation using `.add` or `.cat`.

Different processing modes in `forward` vs `step`:

- **`forward`** processes the whole sequence at once, so pending
  sources live only for the duration of one pipeline walk. The
  "pending" collection is a local data structure inside `forward`.
- **`step` / inference** processes one timestep at a time, so pending
  sources must be carried across `step()` calls as part of the state.
  Concretely, `states` gains extra slots for skip sources currently
  in flight; `initial_step` warms them up during padding exactly the
  same way.

## Implementation shape

- `SkipDef(LayerDef)` — parses `skip.X.(add|cat)` from the spec.
  Carries `X`, the merge op, `__str__`, and `num_weights = 0`.
  `length = 1` (intra-timestep; no time look-back like suffix).
  `is_valid` and `neighbors` are mostly trivial here (`X ≥ 1`, swap
  `add ↔ cat`) — anything that depends on surrounding context lives
  in `ModelDef`.
- `SkipModule(LayerModule)` / `SkipJax(LayerJax)` — no-ops.
  `build_module` / `build_jax` return placeholder modules that
  `Model` / `ModelJax` recognize and handle outside the normal
  per-layer loop.
- `ModelDef` — the real home for skip rules:
  - Validity (no adjacent skips, source-before-norm, end-after-suffix,
    pipeline bounds).
  - Neighbor generation (add/remove skip, distance mutation with
    the suffix skip-over, distance bumps when inserting norm/suffix
    between start and end, etc.).
  - Size computation at merge points.
- `Model` / `ModelJax` — identify skip positions at build time,
  compute merge points, allocate pending-source storage, and handle
  save + merge during `forward`, `step`, and `initial_step`.

This doc will be updated once implementation is complete with any
wrinkles we find along the way.
