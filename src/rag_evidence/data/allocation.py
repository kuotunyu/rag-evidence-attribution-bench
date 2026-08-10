"""Deterministic whole-component allocation for leakage-resistant splits."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from rag_evidence.data.grouping import SplitGroup
from rag_evidence.errors import DataError

_SPLIT_NAMES = ("smoke", "dev", "eval")
_DRAW_ORDER = ("eval", "dev", "smoke")


@dataclass(frozen=True)
class AllocationResult:
    groups: dict[str, tuple[SplitGroup, ...]]
    requested_sizes: dict[str, int]
    realized_sizes: dict[str, int]
    type_distributions: dict[str, dict[str, int]]
    unselected_group_count: int
    unselected_question_count: int


@dataclass(frozen=True)
class _Node:
    previous: int | None
    group_index: int | None


def _validate_inputs(groups: tuple[SplitGroup, ...], requested_sizes: dict[str, int]) -> None:
    if set(requested_sizes) != set(_SPLIT_NAMES):
        raise DataError(f"requested split names must be exactly {_SPLIT_NAMES}")
    if any(size <= 0 for size in requested_sizes.values()):
        raise DataError("requested split sizes must be positive")
    group_ids = [group.group_id for group in groups]
    if len(group_ids) != len(set(group_ids)):
        raise DataError("duplicate group IDs cannot be allocated")
    for group in groups:
        if (
            group.bridge_count < 0
            or group.comparison_count < 0
            or group.bridge_count + group.comparison_count != group.size
        ):
            raise DataError(f"group {group.group_id} type counts do not equal its size")


def _seeded_groups(groups: tuple[SplitGroup, ...], seed: int) -> tuple[SplitGroup, ...]:
    return tuple(
        sorted(
            groups,
            key=lambda group: hashlib.sha256(
                f"{seed}:{group.group_id}".encode()
            ).hexdigest(),
        )
    )


def _build_states(
    ordered: tuple[SplitGroup, ...], *, size_cap: int
) -> tuple[list[_Node], dict[tuple[int, int], int]]:
    """Build reachable (size, bridge-count) states up to an inclusive size cap."""
    nodes = [_Node(previous=None, group_index=None)]
    states: dict[tuple[int, int], int] = {(0, 0): 0}
    max_reachable_states = (size_cap + 1) * (size_cap + 2) // 2
    for group_index, group in enumerate(ordered):
        for (size, bridge_count), node_index in tuple(states.items()):
            new_state = (size + group.size, bridge_count + group.bridge_count)
            if new_state[0] > size_cap or new_state in states:
                continue
            nodes.append(_Node(previous=node_index, group_index=group_index))
            states[new_state] = len(nodes) - 1
        if len(states) == max_reachable_states:
            break
    return nodes, states


def _select_groups(
    available: tuple[SplitGroup, ...],
    *,
    requested_size: int,
    target_bridge_count: int,
    seed: int,
) -> tuple[SplitGroup, ...]:
    if not available:
        return ()
    ordered = _seeded_groups(available, seed)
    nodes, states = _build_states(ordered, size_cap=requested_size)
    if not any(size == requested_size for size, _bridge_count in states):
        reachable_under = [size for size, _bridge_count in states if size > 0]
        if reachable_under:
            nearest_under = max(reachable_under)
            size_cap = requested_size + (requested_size - nearest_under)
        else:
            size_cap = min(group.size for group in ordered)
        nodes, states = _build_states(ordered, size_cap=size_cap)

    nonempty_states = ((state, node) for state, node in states.items() if state[0] > 0)
    (_, _), terminal = min(
        nonempty_states,
        key=lambda item: (
            abs(item[0][0] - requested_size),
            abs(item[0][1] - target_bridge_count),
            item[0][0] > requested_size,
            item[1],
        ),
    )
    selected: list[SplitGroup] = []
    node_index = terminal
    while node_index:
        node = nodes[node_index]
        assert node.group_index is not None
        selected.append(ordered[node.group_index])
        assert node.previous is not None
        node_index = node.previous
    return tuple(sorted(selected, key=lambda group: group.group_id))


def allocate_split_groups(
    groups: tuple[SplitGroup, ...],
    *,
    requested_sizes: dict[str, int],
    seed: int,
) -> AllocationResult:
    """Allocate complete groups in eval/dev/smoke order with deterministic ties."""
    _validate_inputs(groups, requested_sizes)
    pool_size = sum(group.size for group in groups)
    pool_bridge_count = sum(group.bridge_count for group in groups)
    available = tuple(sorted(groups, key=lambda group: group.group_id))
    allocated: dict[str, tuple[SplitGroup, ...]] = {}
    for draw_index, split in enumerate(_DRAW_ORDER):
        target_bridge_count = round(
            requested_sizes[split] * pool_bridge_count / pool_size
        ) if pool_size else 0
        chosen = _select_groups(
            available,
            requested_size=requested_sizes[split],
            target_bridge_count=target_bridge_count,
            seed=seed + draw_index,
        )
        allocated[split] = chosen
        chosen_ids = {group.group_id for group in chosen}
        available = tuple(group for group in available if group.group_id not in chosen_ids)

    groups_by_split = {name: allocated[name] for name in _SPLIT_NAMES}
    realized_sizes = {
        name: sum(group.size for group in rows) for name, rows in groups_by_split.items()
    }
    distributions = {
        name: {
            "bridge": sum(group.bridge_count for group in rows),
            "comparison": sum(group.comparison_count for group in rows),
        }
        for name, rows in groups_by_split.items()
    }
    return AllocationResult(
        groups=groups_by_split,
        requested_sizes=dict(requested_sizes),
        realized_sizes=realized_sizes,
        type_distributions=distributions,
        unselected_group_count=len(available),
        unselected_question_count=sum(group.size for group in available),
    )
