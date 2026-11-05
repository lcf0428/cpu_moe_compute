# CPU/NPU Hybrid MoE Interface (Phase 1)

## Current scope

The first implementation establishes the NPU-side integration points for
single-card decode. It does not implement a CPU expert kernel.

The existing expert offload path remains the default. Hybrid execution is
enabled only after a CPU backend is explicitly attached to a MoE layer.

## Execution flow

1. Routing produces global `topk_ids` and `topk_weights`.
2. `HybridExpertExecutor.prepare()` reads `log2phy[topk_ids]`.
3. Routes whose physical slot is non-negative keep their weight on the NPU.
4. Routes whose physical slot is negative keep their weight on the CPU.
5. The CPU task is submitted before the NPU fused expert operation.
6. The NPU path runs with the original global IDs and NPU-only weights.
7. After NPU `token_combine`, the runner waits for the CPU result and adds it.
8. The merged tensor enters the existing `finalize()` path.

Keeping global expert IDs avoids changing the routing API. Zero route weights
ensure that the NPU placeholder slot contributes nothing for CPU-owned routes.

## CPU backend contract

The backend implements:

```python
submit(
    *,
    layer_idx: int,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> CPUExpertTask
```

`CPUExpertTask.wait()` returns one token-major tensor with shape `[T, H]`.

The backend is responsible for:

- transferring or staging inputs required by the CPU implementation;
- skipping every route whose id is negative;
- executing the selected CPU experts;
- applying each route weight exactly once;
- reducing all CPU expert contributions for each token;
- returning a tensor compatible with the routed NPU output.

The backend takes **no** resident-expert mask of its own.  The CPU/NPU split
has exactly one source of truth -- `log2phy`, read by
`HybridExpertExecutor.prepare()` on every step.  kt-kernel's
`should_skip_expert` skips a negative id unconditionally, so the `-1` sentinel
already covers every route a mask would have covered, and it is re-derived
per step rather than frozen at construction time.  A second, construction-time
mask is therefore not merely redundant: when it drifts from `log2phy`, both
sides skip the route and the expert is dropped silently -- no error, no crash,
just a numerically smaller output on the affected tokens.

The NPU integration converts the returned tensor to the NPU output device and
dtype before addition. A production backend should normally manage pinned
buffers and asynchronous copies itself to avoid a blocking hot path.

## Activation

Attach a concrete backend during layer/model initialization:

```python
from vllm_ascend.expert_offload.hybrid_executor import (
    attach_hybrid_expert_executor,
)

attach_hybrid_expert_executor(
    layer,
    backend=cpu_backend,
    layer_idx=layer_idx,
)
```

Without this attachment, the original weight-paging implementation continues
to call `ExpertOffloadManager.update_weights()`.

## Current integration boundary

The runtime hook is currently wired into the W4A8-MXFP4 single-card decode
path. Prefill, multi-card execution, and other quantization methods retain the
existing behavior until dedicated integrations are added.
