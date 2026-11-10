"""Short H100 component comparisons against --baseline-ref.

Run against a candidate PR checkout on free benchmark GPUs. Measures native
WO-A + consumer layout, DPA8 collectives, or the full native MXFP4 MoE wrapper.
Tensor values/routing are synthetic at actual model dimensions. These timings
are not an end-to-end serving or model-quality benchmark.
"""

import __future__

import argparse
import ast
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist

SEED = 66068407
BASE = "eb73a50b1d6183deb1d79ca425677bd0cf93342f"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"


def baseline_function(module, relative, name):
    source = Path(module.__file__).resolve()
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=source.parent, text=True
        ).strip()
    )
    parent = subprocess.check_output(
        ["git", "rev-parse", BASE], cwd=root, text=True
    ).strip()
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", parent, "HEAD"], cwd=root, check=True
    )
    text = subprocess.check_output(
        ["git", "show", f"{parent}:{relative}"], cwd=root, text=True
    )
    node = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )
    assert not node.decorator_list, (
        "preserve decorators before benchmarking this control"
    )
    definition = ast.get_source_segment(text, node)
    namespace = dict(module.__dict__)
    exec(
        compile(
            definition,
            f"<exact-parent:{relative}:{name}>",
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace[name], hashlib.sha256(definition.encode()).hexdigest()


class Bench:
    def __init__(self, output, group, unroll, launches, repeats):
        self.output, self.group = output, group
        self.unroll, self.launches, self.repeats = unroll, launches, repeats
        self.rank = dist.get_rank() if group is not None else 0
        self.rows = []

    def barrier(self):
        if self.group is not None:
            dist.barrier(group=self.group.cpu_group)

    def capture(self, function):
        graph = torch.cuda.CUDAGraph()
        if self.group is None:
            with torch.cuda.graph(graph):
                for _ in range(self.unroll):
                    out = function()
        else:
            with self.group.graph_capture() as capture:
                with torch.cuda.graph(graph, stream=capture.stream):
                    for _ in range(self.unroll):
                        out = function()
        return graph, out

    def pair(self, case, baseline, candidate, check):
        for _ in range(20):
            baseline()
            candidate()
        torch.cuda.synchronize()
        check(baseline(), candidate())
        graphs = [self.capture(fn) for fn in (baseline, candidate)]
        for index in (0, 1, 1, 0):
            for _ in range(20):
                graphs[index][0].replay()
        torch.cuda.synchronize()
        check(graphs[0][1], graphs[1][1])
        samples = []
        for repeat in range(self.repeats):
            order = (0, 1) if repeat % 2 == 0 else (1, 0)
            sample = {"repeat": repeat, "order": order}
            for arm in order:
                self.barrier()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(self.launches):
                    graphs[arm][0].replay()
                end.record()
                end.synchronize()
                local = start.elapsed_time(end) * 1000 / (self.launches * self.unroll)
                rank_values = [local]
                if self.group is not None:
                    rank_values = [None] * dist.get_world_size(
                        group=self.group.cpu_group
                    )
                    dist.all_gather_object(
                        rank_values, local, group=self.group.cpu_group
                    )
                sample["baseline_us" if arm == 0 else "candidate_us"] = max(rank_values)
                sample[
                    "baseline_per_rank_us" if arm == 0 else "candidate_per_rank_us"
                ] = rank_values
            samples.append(sample)
        result = {
            "case": case,
            "samples": samples,
            "baseline_median_us": statistics.median(s["baseline_us"] for s in samples),
            "candidate_median_us": statistics.median(
                s["candidate_us"] for s in samples
            ),
            "paired_speedup_median": statistics.median(
                s["baseline_us"] / s["candidate_us"] for s in samples
            ),
            "unroll": self.unroll,
            "launches": self.launches,
            "repeats": self.repeats,
            "timing": "CUDA events around graph replays / operations, max across rank times, alternating AB/BA",
            "correctness_checked": True,
        }
        self.rows.append(result)
        if self.rank == 0:
            with (self.output / "journal.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            print(
                json.dumps({k: v for k, v in result.items() if k != "samples"}),
                flush=True,
            )
        del graphs
        torch.cuda.synchronize()
        return result


def init_distributed():
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.layers import dp_attention as dp
    from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish
    from sglang.srt.server_args import ServerArgs

    rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 8
    torch.cuda.set_device(rank)
    args = ServerArgs(
        model_path=MODEL,
        trust_remote_code=True,
        device="cuda",
        tp_size=8,
        attn_dp_size=8,
        ep_size=8,
        attn_cp_size=1,
        attention_backend="dsv4",
        moe_runner_backend="flashinfer_mxfp4",
        flashinfer_mxfp4_moe_precision="fp8",
        speculative_algorithm="DSPARK",
        speculative_dspark_block_size=5,
        enable_dp_lm_head=True,
    )
    publish(args, role="test", ranks=SpawnRanks(world_rank=rank, gpu_id=rank))
    ps.init_distributed_environment(
        world_size=8, rank=rank, local_rank=rank, backend="nccl", timeout=180
    )
    ps.initialize_model_parallel()
    dp.initialize_dp_attention(args)
    dp.set_is_extend_in_batch(False)
    return get_parallel().tp_group


def woa(bench):
    from sglang.srt.models import deepseek_v4 as module

    parent, sha = baseline_function(
        module, "python/sglang/srt/models/deepseek_v4.py", "_apply_wo_a_bf16_matmul"
    )
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    w = (
        torch.randn((8, 1024, 4096), generator=gen, device="cuda", dtype=torch.bfloat16)
        * 0.02
    )
    for rows, strided in (
        (1, False),
        (6, False),
        (6, True),
        (12, False),
        (24, False),
        (48, False),
        (49, False),
    ):
        storage = (
            torch.randn(
                (rows, 16 if strided else 8, 4096),
                generator=gen,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.1
        )
        x = storage[:, :8]
        kwargs = dict(is_decode=False, is_target_verify=True, fast_path=True)

        def before():
            return parent(x, w, **kwargs).flatten(1)

        def after():
            return module._apply_wo_a_bf16_matmul(x, w, **kwargs).flatten(1)

        def check(a, b):
            torch.testing.assert_close(a, b, rtol=2**-6, atol=2**-10)

        with mock.patch.object(
            module,
            "_wo_a_bf16_bmm_token_major",
            wraps=module._wo_a_bf16_bmm_token_major,
        ) as routed:
            after()
            dispatch_hits = routed.call_count
        assert dispatch_hits == (1 if rows <= 48 else 0)
        bench.pair(
            {
                "dispatch": {
                    "native_token_major_bmm_calls": dispatch_hits,
                    "intentional_fallback": rows == 49,
                },
                "component": "wo-a",
                "rows": rows,
                "strided": strided,
                "weight": list(w.shape),
                "activation_stride": list(x.stride()),
                "seam": "BF16 WO-A plus flatten(1) required by WO-B caller",
                "parent_function_sha256": sha,
            },
            before,
            after,
            check,
        )


def collectives(bench):
    from sglang.srt.layers import dp_attention as dp

    group = bench.group
    parent_gather, sha = baseline_function(
        dp, "python/sglang/srt/layers/dp_attention.py", "_dp_gather_via_all_gather"
    )
    rank = bench.rank
    for rows in (1, 5, 6, 12, 24, 48):
        local = (
            torch.arange(rows * 5120, device="cuda").reshape(rows, 5120) % 5 + rank
        ).to(torch.bfloat16)
        old, new = (
            torch.empty((rows * 8, 5120), device="cuda", dtype=torch.bfloat16),
            torch.empty((rows * 8, 5120), device="cuda", dtype=torch.bfloat16),
        )
        source = (
            torch.arange(rows * 8 * 5120, device="cuda").reshape(rows * 8, 5120) % 5
            + rank
        ).to(torch.bfloat16)
        old_r, new_r = torch.empty_like(local), torch.empty_like(local)
        sizes = [rows] * 8

        def old_gather():
            parent_gather(old, local, None, True)
            return old

        def new_gather():
            dp._dp_gather_via_all_gather(new, local, None, True)
            return new

        def old_reduce():
            group.reduce_scatterv(source, output=old_r, sizes=sizes)
            return old_r

        def new_reduce():
            dp.dp_reduce_scatterv(new_r, source, sizes)
            return new_r

        def check(a, b):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

        from sglang.kernels.ops.communication import nvlink_comm

        with mock.patch.object(
            nvlink_comm, "all_gather_push", wraps=nvlink_comm.all_gather_push
        ) as routed:
            new_gather()
            gather_hits = routed.call_count
        assert gather_hits == 1
        with mock.patch.object(
            group, "reduce_scatterv", wraps=group.reduce_scatterv
        ) as routed:
            new_reduce()
            assert routed.call_count == 1 and routed.call_args.kwargs["sizes"] is None
        dispatch = {
            "native_nvlink_push_calls": gather_hits,
            "native_reduce_scatter_sizes": None,
        }
        dispatch_by_rank = [None] * 8
        dist.all_gather_object(dispatch_by_rank, dispatch, group=group.cpu_group)
        for name, before, after in (
            ("gather", old_gather, new_gather),
            ("reduce_scatter", old_reduce, new_reduce),
        ):
            bench.pair(
                {
                    "dispatch_by_rank": dispatch_by_rank,
                    "component": "collectives",
                    "operation": name,
                    "rows_per_rank": rows,
                    "hidden": 5120,
                    "ranks": 8,
                    "dtype": "bfloat16",
                    "parent_gather_sha256": sha,
                },
                before,
                after,
                check,
            )


def padding(bench):
    from flashinfer.fused_moe import preprocess_moe_weights_for_sm90_mixed_gemm_humming

    from sglang.srt.layers import dp_attention as dp
    from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass as module
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    parent, sha = baseline_function(
        module,
        "python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py",
        "_fused_experts_flashinfer_mxfp4_cutlass",
    )
    rank = bench.rank
    gen = torch.Generator(device="cuda").manual_seed(SEED + rank)
    hidden, intermediate, experts = 5120, 2304, 48
    w13 = torch.randint(
        0,
        256,
        (experts, 2 * intermediate, hidden // 2),
        device="cuda",
        dtype=torch.uint8,
        generator=gen,
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden, intermediate // 2),
        device="cuda",
        dtype=torch.uint8,
        generator=gen,
    )
    s13 = torch.full(
        (experts, 2 * intermediate, hidden // 32), 120, device="cuda", dtype=torch.uint8
    )
    s2 = torch.full(
        (experts, hidden, intermediate // 32), 120, device="cuda", dtype=torch.uint8
    )

    # FlashInfer's eager payload rewrite expands indices to int64. Prepare
    # independent experts separately; this changes only untimed fixture prep.
    def prepare(weight, scale):
        parts = [
            preprocess_moe_weights_for_sm90_mixed_gemm_humming(
                weight[i : i + 1], scale[i : i + 1]
            )
            for i in range(weight.shape[0])
        ]
        return tuple(torch.cat([part[k] for part in parts], dim=0) for k in range(3))

    for weight, scale in ((w13, s13), (w2, s2)):
        probe_scale = scale[:2].clone()
        probe_scale[1].add_(1)
        probe_scale[0, 0, 0] = 106  # Exercise the per-expert range clamp.
        full = preprocess_moe_weights_for_sm90_mixed_gemm_humming(
            weight[:2], probe_scale
        )
        chunked = prepare(weight[:2], probe_scale)
        for a, b in zip(full, chunked):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        del full, chunked, probe_scale
    w13, s13, r13 = prepare(w13, s13)
    w2, s2, r2 = prepare(w2, s2)
    quant = module.FlashInferCutlassMxfp4MoeQuantInfo(
        w13_weight=w13,
        w2_weight=w2,
        w13_weight_scale=s13,
        w2_weight_scale=s2,
        w13_humming_residual_scale=(r13 * 64).contiguous(),
        w2_humming_residual_scale=(r2 * 64).contiguous(),
        humming_fc2_act_scale=torch.ones((), device="cuda", dtype=torch.float32),
    )
    layer = SimpleNamespace(
        moe_tp_size=1, moe_tp_rank=0, moe_ep_size=8, moe_ep_rank=rank
    )
    config = MoeRunnerConfig(
        layer=layer,
        num_experts=384,
        num_local_experts=48,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        top_k=6,
    )
    for active in (
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1] * 8,
        [4, 1, 1, 0, 0, 0, 0, 0],
        [4, 3, 2, 2, 2, 2, 1, 1],
        [8, 2, 1, 1, 1, 0, 0, 0],
        [8] * 8,
    ):
        per_rank = max(active) * 6
        rows = 8 * per_rank
        counts = torch.tensor(active, device="cuda", dtype=torch.int32) * 6
        dp.set_dp_buffer_len(
            rows, per_rank, True, [per_rank] * 8, moe_dp_valid_counts=counts
        )
        x = (
            torch.randn(
                (rows, hidden), device="cuda", dtype=torch.bfloat16, generator=gen
            )
            * 0.1
        )
        routes = (
            (
                torch.arange(rows, device="cuda")[:, None] * 13
                + torch.arange(6, device="cuda")[None, :] * 37
            )
            % 384
        ).to(torch.int32)
        original = routes.clone()
        dispatch = StandardDispatchOutput(
            hidden_states=x,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=torch.full(
                    (rows, 6), 1 / 6, device="cuda", dtype=torch.float32
                ),
                topk_ids=routes,
                router_logits=torch.empty((rows, 384), device="cuda"),
            ),
        )
        valid = torch.arange(rows, device="cuda") % per_rank < counts.repeat_interleave(
            per_rank
        )

        def before():
            return parent(dispatch, quant, config).hidden_states

        def after():
            return module._fused_experts_flashinfer_mxfp4_cutlass(
                dispatch, quant, config
            ).hidden_states

        def check(a, b):
            torch.testing.assert_close(a[valid], b[valid], rtol=2**-6, atol=2**-8)
            torch.testing.assert_close(
                b[~valid], torch.zeros_like(b[~valid]), rtol=0, atol=0
            )
            torch.testing.assert_close(routes, original, rtol=0, atol=0)
            assert torch.isfinite(a).all() and torch.isfinite(b).all()

        with mock.patch.object(
            dp, "copy_mask_dp_moe_padding", wraps=dp.copy_mask_dp_moe_padding
        ) as routed:
            after()
            assert routed.call_count == 1
        dispatch_by_rank = [None] * 8
        dist.all_gather_object(
            dispatch_by_rank, {"native_copy_mask_calls": 1}, group=bench.group.cpu_group
        )
        bench.pair(
            {
                "dispatch_by_rank": dispatch_by_rank,
                "component": "padding",
                "active_requests_per_rank": active,
                "captured_requests_per_rank": max(active),
                "verify_tokens_per_request": 6,
                "valid_fraction": sum(active) / (max(active) * 8),
                "hidden": hidden,
                "intermediate": intermediate,
                "local_experts": experts,
                "global_experts": 384,
                "routing": "six distinct global experts, balanced cyclic assignment",
                "seam": "complete native MoE wrapper including copy/mask, activation prep, Humming/CUTLASS and finalization",
                "parent_function_sha256": sha,
            },
            before,
            after,
            check,
        )


def main():
    global BASE, MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component", choices=("wo-a", "collectives", "padding"), required=True
    )
    parser.add_argument("--baseline-ref", default=BASE)
    parser.add_argument("--model-path", default=MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unroll", type=int, default=8)
    parser.add_argument("--launches", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    BASE, MODEL = args.baseline_ref, args.model_path
    assert args.unroll > 0 and args.launches > 0 and args.repeats >= 4
    if int(os.environ.get("RANK", "0")) == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        assert not any(
            (args.output / name).exists()
            for name in ("contract.json", "complete.json", "journal.jsonl")
        ), "use a fresh output directory"
    group = init_distributed() if args.component != "wo-a" else None
    assert torch.cuda.get_device_capability() == (9, 0)
    bench = Bench(args.output, group, args.unroll, args.launches, args.repeats)
    source = Path(__import__("sglang").__file__).resolve().parents[2]
    BASE = subprocess.check_output(
        ["git", "rev-parse", BASE], cwd=source, text=True
    ).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    assert not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=source, text=True
    )
    initial = {
        "component": args.component,
        "commit": commit,
        "base": BASE,
        "at": time.time(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "hardware": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,memory.free,utilization.gpu,clocks.sm",
                "--format=csv,noheader",
            ],
            text=True,
        ).splitlines(),
        "preparation_timed": False,
        "scope": "native component timing, synthetic tensor values at real dimensions, not end-to-end serving or task quality",
    }
    if bench.rank == 0:
        (args.output / "contract.json").write_text(json.dumps(initial, indent=2))
    with torch.inference_mode():
        {"wo-a": woa, "collectives": collectives, "padding": padding}[args.component](
            bench
        )
    if bench.rank == 0:
        (args.output / "complete.json").write_text(
            json.dumps(
                {
                    **initial,
                    "ended_at": time.time(),
                    "passed": True,
                    "cases": len(bench.rows),
                    "job_stopped": False,
                },
                indent=2,
            )
        )
    if group is not None:
        dist.barrier(group=group.cpu_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
