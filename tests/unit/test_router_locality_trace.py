import math

from lqh.experiments.router_locality.trace import summarize_selected_experts


def test_summarize_selected_experts_counts_union_and_churn() -> None:
    record = summarize_selected_experts(
        [[0, 1], [0, 1], [1, 2]], layer=7, sequence_index=3
    )

    assert record.layer == 7
    assert record.sequence_index == 3
    assert record.tokens == 3
    assert record.top_k == 2
    assert record.unique_experts == 3
    assert record.expert_selection_counts == {0: 2, 1: 3, 2: 1}
    assert math.isclose(record.mean_adjacent_jaccard_distance, 1 / 3)


def test_summarize_selected_experts_single_token_has_no_churn() -> None:
    record = summarize_selected_experts([[4, 9]], layer=0, sequence_index=0)

    assert record.mean_adjacent_jaccard_distance == 0.0
