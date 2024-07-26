# Inspired from https://github.com/zyushun/Adam-mini/blob/bacb397866a1f8bbcb827f5d6d31b7e255f11c98/adam_mini/adam_mini.py.

from typing import Any, Callable, Iterable, Optional
from functools import partial

import torch

from nanotron.distributed import ProcessGroup
from nanotron.config.models_config import NanotronConfigs, LlamaConfig


class AdamMini(torch.optim.Optimizer):
    def __init__(
            self,
            param_groups: list[dict[str, Any]],
            tp_pg: ProcessGroup,
            lr: float = 1e-3,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0,
    ):

        self.tp_pg = tp_pg
        defaults = {"lr": lr, "beta1": betas[0], "beta2": betas[1], "eps": eps,
                    "weight_decay": weight_decay}
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if "splitter" in group:
                    param_views = group["splitter"](param)
                    grad_views = group["splitter"](param.grad)
                    if "states" not in self.state[param]:
                        self.state[param]["states"] = [{} for _ in range(len(param_views))]
                    it = enumerate(zip(param_views, grad_views, group["dense"], group["unshardable"]))
                    for i, (param_view, grad_view, dense, unshardable) in it:
                        param_view.grad = grad_view  # We need to manually repopulate the gradient.
                        adam_mini_step(param_view, self.state[param]["states"][i], self.tp_pg, dense, unshardable,
                                       group["lr"], group["beta1"], group["beta2"], group["eps"], group["weight_decay"])
                else:
                    adam_mini_step(param, self.state[param], self.tp_pg, group["dense"], group["unshardable"],
                                   group["lr"], group["beta1"], group["beta2"], group["eps"], group["weight_decay"])
        return loss


def llama_qkv_splitter(n_local_q_heads: int, n_local_kv_heads: int, d_qk: int,
                       qkv_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    qk_weight = qkv_weight[n_local_q_heads*d_qk + n_local_kv_heads*d_qk :, :]
    v_weight = qkv_weight[: n_local_q_heads*d_qk + n_local_kv_heads*d_qk, :]

    qk_weight = qk_weight.view(n_local_q_heads + n_local_kv_heads, -1)
    v_weight = v_weight.view(n_local_kv_heads, -1)
    return qk_weight, v_weight


def adam_mini_step(param: torch.Tensor, state: dict[str, Any], tp_pg: ProcessGroup,
                   dense: bool, unshardable: bool, lr: float, beta1: float, beta2: float,
                   eps: float, weight_decay: float):

    if param.grad is None:
        return
    unshardable = unshardable or tp_pg.size() == 1
    grad = param.grad.to(torch.float32)

    # Initialize state, if needed.
    if state == {}:
        state["m"] = torch.zeros_like(param, dtype=torch.float32)
        state["step"] = 0
        if dense == "row":
            state["vrow"] = torch.zeros(param.size(0), 1, dtype=torch.float32, device=param.device)
        elif dense:
            state["v"] = torch.zeros_like(param, dtype=torch.float32)
        else:
            state["vmean"] = torch.zeros([], dtype=torch.float32, device=param.device)
            if not unshardable:
                numel = torch.tensor(param.numel(), dtype=torch.float32, device=param.device)
                torch.distributed.all_reduce(numel, group=tp_pg)
                assert numel > param.numel()  # Otherwise the other ranks wouldn't have any parameters...
                state["numel"] = numel.item()

    # TODO: we might want to async the all_reduces.
    # Modify either v or v_mean, depending on the density.
    if dense == "row":
        assert unshardable
        tmp_lr = torch.mean(grad*grad, dim=1, keepdim=True)
        state["vrow"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
        v_or_vmean = state["vrow"]
    elif dense:
        assert unshardable
        state["v"].mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        v_or_vmean = state["v"]
    else:
        if unshardable:
            tmp_lr = torch.mean(grad*grad)
        else:
            tmp_lr = torch.sum(grad*grad)
            torch.distributed.all_reduce(tmp_lr, op=torch.distributed.ReduceOp.SUM)
            tmp_lr = tmp_lr/state["numel"]
        state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
        v_or_vmean = state["vmean"]

    # Now we can proceed with every case in almost the same way.
    state["step"] += 1
    if weight_decay > 0.0:
        param.mul_(1 - lr*weight_decay)
    state["m"].lerp_(grad, 1 - beta1)
    bias_correction_1 = 1 - beta1**state["step"]
    bias_correction_2 = 1 - beta2**state["step"]
    # h = sqrt(v/corr2) + eps
    h = torch.sqrt(v_or_vmean).div_(bias_correction_2**0.5).add_(eps)
    #h = (torch.sqrt(v_or_vmean)/bias_correction_2**0.5).add_(eps)

    # Apply the final update to param, using specialized functions for each case.
    # param = param - lr * m/(h*corr1)
    if dense:  # dense=True and dense=row case are equal because h is broadcastable.
        param.addcdiv_(state["m"], h, value=-lr/bias_correction_1)
    else:
        param.add_(state["m"], alpha=-lr/h.mul_(bias_correction_1))


def llama_partitioner(config: LlamaConfig, tp_pg: ProcessGroup, default_groups: Iterable[dict[str, Any]],
                      id_to_name: dict[int, str]) -> list[dict[str, Any]]:

    n_local_q_heads = config.num_attention_heads // tp_pg.size()
    n_local_kv_heads = config.num_key_value_heads // tp_pg.size()
    d_qk = config.hidden_size // config.num_attention_heads

    new_groups = []
    for group in default_groups:
        for param in group["params"]:
            name = id_to_name[id(param)]
            overrides = {key: value for key, value in group.items() if key != "params"}  # Copy the chosen settings of this group.
            overrides["params"] = [param]

            # For token_embeddings and lm_heads, even if they are effectively sharded when tp>1, we still
            # set unshardable to True. This is because both parameters are "dense", so we will use the full
            # grad*grad (instead of the reduced mean(grad*grad)) as state, so no communication between tp ranks is needed.
            if "token_embedding" in name:
                new_groups.append({**overrides, "dense": True, "unshardable": True})
            elif "lm_head" in name:
                new_groups.append({**overrides, "dense": True, "unshardable": True})

            # Layernorm parameters are not sharded, so each tp rank has its own full copy of the grad and
            # parameter. No communication between ranks is needed to accurately compute mean(grad*grad).
            elif "layernorm" in name or "layer_norm" in name:
                new_groups.append({**overrides, "dense": False, "unshardable": True})

            # The case of key, query and value weights is tricky because we need to, first, separate the
            # value weights from the query and key.
            # And also, each query and key head is a separate parameter group, so we need to separate those also.
            # We set the query and values to unshardable because every head corresponds to one group, and we don't shard
            # each individual head, but rather across heads.
            # That's why value is not unshardable.
            # TODO: Update above text.
            # TODO: Paper says that the values don't need head-wise density, but it might be worth it.
            elif "qkv_proj" in name:
                #new_groups.append({**overrides, "dense": ("row", False), "unshardable": (True, False),
                new_groups.append({**overrides, "dense": ("row", "row"), "unshardable": (True, True),
                                   "splitter": partial(llama_qkv_splitter, n_local_q_heads, n_local_kv_heads, d_qk)})

                ## We separate Q,K,V and add the value group.
                #sizes = (n_local_q_heads*d_qk, n_local_kv_heads*d_qk, n_local_kv_heads*d_qk)
                #query_weight, key_weight, value_weight = torch.split(param, sizes)
                #new_groups.append({**overrides, "dense": False, "unshardable": False, "params": [value_weight]})

                ## Now we can add the Q,K parameters separately.
                #partition = list(torch.split(query_weight, d_qk)) + list(torch.split(key_weight, d_qk))
                #assert len(partition) == n_local_q_heads + n_local_kv_heads
                #new_groups.append({**overrides, "dense": False, "unshardable": True, "params": partition})

            # Finally, all other weights will not use any special partition, or density, and all are sharded across tp.
            else:
                new_groups.append({**overrides, "dense": False, "unshardable": False})

    assert {param for group in new_groups for param in group["params"]} == {param for group in default_groups for param in group["params"]}
    return new_groups


def get_param_partitioner(config: NanotronConfigs, tp_pg: ProcessGroup) \
                          -> Callable[[Iterable[dict[str, Any]], dict[int, str]], Iterable[dict[str, Any]]]:
    # TODO: Document splitter.
    # TODO: Document "row" density.
    """
    Each parameter group contains at least these three keys:
        parameters (list[torch.Tensor]): The parameters of the parameter group.
        dense (bool):
            If true, the parameters of this group will have dense second-moment state (akin to original AdamW),
            otherwise we use the mean() of the second-moment state for the entire parameter.
            This is only used in the token embeddings and output layer.
        unshardable (bool):
            If true, the parameters will not be ever sharded, even when using tp>1.
            Otherwise, it is understood that when tp>1, in order to compute the true mean of `grad*grad`,
            one must all_reduce the local `sum(grad*grad) first, find the global `numel` and then compute `all_reduced_grad_x_grad/numel`.
    """
    if isinstance(config, LlamaConfig):
        assert not config.rope_interleaved, "Llama must not use interleaved roped when using AdamMini"
        return partial(llama_partitioner, config, tp_pg)
    raise NotImplementedError(f"AdamMini parameter partition for models {config} not yet implemented")
