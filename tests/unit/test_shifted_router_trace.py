import math

from lqh.experiments.shifted_router.trace import summarize_prefetch


def test_summarize_prefetch_perfect_agreement() -> None:
    real = [[0, 1], [2, 3]]
    shifted = [[0, 1], [2, 3]]

    record = summarize_prefetch(real, shifted, layer=5, sequence_index=1)

    assert record.layer == 5
    assert record.sequence_index == 1
    assert record.tokens == 2
    assert record.top_k == 2
    assert record.mean_hit_rate == 1.0
    assert record.mean_wasted_rate == 0.0
    assert record.real_unique_experts == 4
    assert record.shifted_unique_experts == 4
    assert record.shifted_extra_unique_experts == 0


def test_summarize_prefetch_no_agreement() -> None:
    real = [[0, 1]]
    shifted = [[2, 3]]

    record = summarize_prefetch(real, shifted, layer=0, sequence_index=0)

    assert record.mean_hit_rate == 0.0
    assert record.mean_wasted_rate == 1.0
    assert record.shifted_extra_unique_experts == 2


def test_summarize_prefetch_partial_overlap() -> None:
    # real picks {0,1}, shifted guesses {0,2}: 1 of 2 real experts prefetched,
    # 1 of 2 prefetched experts wasted (not actually needed).
    real = [[0, 1]]
    shifted = [[0, 2]]

    record = summarize_prefetch(real, shifted, layer=3, sequence_index=2)

    assert math.isclose(record.mean_hit_rate, 0.5)
    assert math.isclose(record.mean_wasted_rate, 0.5)
    assert record.real_unique_experts == 2
    assert record.shifted_unique_experts == 2
    assert record.shifted_extra_unique_experts == 1  # expert 2 not in real union
