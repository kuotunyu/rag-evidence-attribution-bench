"""Whole leakage groups are allocated deterministically and never split."""

from __future__ import annotations

import pytest

from rag_evidence.data.allocation import allocate_split_groups
from rag_evidence.data.grouping import SplitGroup
from rag_evidence.errors import DataError


def _group(name: str, size: int, bridge_count: int) -> SplitGroup:
    return SplitGroup(
        group_id=name,
        question_ids=tuple(f"{name}-q{i}" for i in range(size)),
        normalized_titles=(f"title-{name}",),
        paragraph_hashes=(f"paragraph-{name}",),
        bridge_count=bridge_count,
        comparison_count=size - bridge_count,
    )


def test_allocator_is_order_independent_and_never_reuses_a_group() -> None:
    groups = (
        _group("g1", 4, 2),
        _group("g2", 3, 2),
        _group("g3", 2, 1),
        _group("g4", 2, 1),
        _group("g5", 1, 1),
        _group("g6", 1, 0),
    )
    sizes = {"smoke": 2, "dev": 3, "eval": 4}

    result = allocate_split_groups(groups, requested_sizes=sizes, seed=17)
    reversed_result = allocate_split_groups(
        tuple(reversed(groups)), requested_sizes=sizes, seed=17
    )

    assert result == reversed_result
    assert result.requested_sizes == sizes
    assert result.realized_sizes == sizes
    selected = [group.group_id for rows in result.groups.values() for group in rows]
    assert len(selected) == len(set(selected))
    assert result.unselected_group_count == len(groups) - len(selected)
    assert result.unselected_question_count == 4
    for split, rows in result.groups.items():
        distribution = result.type_distributions[split]
        assert distribution["bridge"] + distribution["comparison"] == sum(
            group.size for group in rows
        )


def test_allocator_uses_nearest_nonempty_whole_group_when_exact_size_is_impossible() -> None:
    groups = tuple(_group(f"g{i}", 3, 2) for i in range(3))

    result = allocate_split_groups(
        groups,
        requested_sizes={"smoke": 1, "dev": 1, "eval": 4},
        seed=5,
    )

    assert result.realized_sizes == {"smoke": 3, "dev": 3, "eval": 3}
    assert all(len(rows) == 1 for rows in result.groups.values())


def test_allocator_prefers_the_closest_bridge_distribution_after_size() -> None:
    balanced = _group("balanced", 2, 1)
    all_bridge = _group("all-bridge", 2, 2)
    all_comparison = _group("all-comparison", 2, 0)
    fillers = (_group("f1", 1, 1), _group("f2", 1, 0))

    result = allocate_split_groups(
        (balanced, all_bridge, all_comparison, *fillers),
        requested_sizes={"smoke": 1, "dev": 1, "eval": 2},
        seed=11,
    )

    assert [group.group_id for group in result.groups["eval"]] == ["balanced"]


def test_seed_breaks_objective_ties_without_breaking_determinism() -> None:
    groups = tuple(_group(f"g{i}", 1, 1) for i in range(8))
    sizes = {"smoke": 1, "dev": 1, "eval": 1}

    selections = {
        tuple(
            group.group_id
            for split in ("eval", "dev", "smoke")
            for group in allocate_split_groups(groups, requested_sizes=sizes, seed=seed).groups[
                split
            ]
        )
        for seed in range(10)
    }

    assert len(selections) > 1


@pytest.mark.parametrize(
    ("groups", "sizes", "message"),
    [
        (
            (_group("dup", 1, 1), _group("dup", 1, 1)),
            {"smoke": 1, "dev": 1, "eval": 1},
            "duplicate",
        ),
        ((_group("bad", 2, 3),), {"smoke": 1, "dev": 1, "eval": 1}, "type counts"),
        ((_group("ok", 1, 1),), {"smoke": 1, "dev": 1}, "split names"),
    ],
)
def test_allocator_rejects_invalid_group_accounting(
    groups: tuple[SplitGroup, ...], sizes: dict[str, int], message: str
) -> None:
    with pytest.raises(DataError, match=message):
        allocate_split_groups(groups, requested_sizes=sizes, seed=1)
