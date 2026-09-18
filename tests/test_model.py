"""Unit tests for the DualEmbeddingChGKModel PyTorch architecture."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from chgk_rating.model import DualEmbeddingChGKModel


def test_model_initialization() -> None:
    num_players = 50
    num_editors = 20
    dim_k = 16
    dim_c = 8

    model = DualEmbeddingChGKModel(
        num_players=num_players,
        num_editors=num_editors,
        dim_k=dim_k,
        dim_c=dim_c,
    )

    assert model.player_k_emb.weight.shape == (num_players, dim_k)
    assert model.player_c_base.weight.shape == (num_players, dim_c)
    assert model.player_c_delta.weight.shape == (num_players, dim_c)
    assert model.editor_emb.weight.shape == (num_editors, dim_k)
    assert model.editor_b.weight.shape == (num_editors, 1)
    assert model.W_synergy.shape == (dim_c, dim_c)

    # Padding index 0 must be strictly zero
    assert torch.all(model.player_k_emb.weight[0] == 0.0)
    assert torch.all(model.player_c_base.weight[0] == 0.0)
    assert torch.all(model.player_c_delta.weight[0] == 0.0)
    assert torch.all(model.editor_emb.weight[0] == 0.0)
    assert torch.all(model.editor_b.weight[0] == 0.0)


def test_forward_pass_shapes_and_values() -> None:
    batch_size = 4
    k_players = 6
    num_players = 30
    num_editors = 15

    model = DualEmbeddingChGKModel(
        num_players=num_players,
        num_editors=num_editors,
        dim_k=8,
        dim_c=4,
    )

    player_ids = torch.tensor([
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 0, 0],  # partial roster with padding
        [11, 12, 13, 14, 15, 16],
        [17, 18, 19, 20, 21, 22],
    ], dtype=torch.long)

    editor_ids = torch.tensor([[1, 2], [3, 0], [4, 5], [0, 0]], dtype=torch.long)
    format_ids = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    out = model(player_ids=player_ids, editor_ids=editor_ids, format_ids=format_ids)

    assert out["logits"].shape == (batch_size, 1)
    assert out["probs"].shape == (batch_size, 1)
    assert out["p_gen"].shape == (batch_size, 1)
    assert out["z_know"].shape == (batch_size, 1)
    assert out["s_team"].shape == (batch_size, 1)
    assert out["z_anomaly"].shape == (batch_size, 1)

    # Probs must be in [0, 1]
    assert torch.all(out["probs"] >= 0.0)
    assert torch.all(out["probs"] <= 1.0)
    assert torch.all(out["p_gen"] >= 0.0)
    assert torch.all(out["p_gen"] <= 1.0)


def test_team_knowledge_pooling() -> None:
    model = DualEmbeddingChGKModel(
        num_players=10,
        num_editors=5,
        dim_k=4,
        dim_c=2,
    )

    # Manually set knowledge embeddings
    with torch.no_grad():
        model.player_k_emb.weight[1] = torch.tensor([1.0, 0.0, 5.0, -2.0])
        model.player_k_emb.weight[2] = torch.tensor([2.0, -1.0, 3.0, 4.0])
        model.player_k_emb.weight[0] = torch.tensor([0.0, 0.0, 0.0, 0.0])  # padding

    player_ids = torch.tensor([[1, 2, 0, 0, 0, 0]], dtype=torch.long)
    editor_ids = torch.tensor([[1]], dtype=torch.long)
    format_ids = torch.tensor([0], dtype=torch.long)

    # Team max pooling should be max(k_1, k_2): [2.0, 0.0, 5.0, 4.0]
    expected_k_team = torch.tensor([[2.0, 0.0, 5.0, 4.0]])

    player_mask = player_ids != 0
    k_player_embs = model.player_k_emb(player_ids)
    masked_k = k_player_embs.masked_fill(~player_mask.unsqueeze(-1), -1e9)
    k_team = torch.max(masked_k, dim=1).values

    assert torch.allclose(k_team, expected_k_team)


def test_pairwise_synergy_computation() -> None:
    model = DualEmbeddingChGKModel(
        num_players=10,
        num_editors=5,
        dim_k=4,
        dim_c=2,
    )

    # Setup known communication embeddings and synergy tensor
    with torch.no_grad():
        model.player_c_base.weight[1] = torch.tensor([1.0, 0.0])
        model.player_c_base.weight[2] = torch.tensor([0.0, 1.0])
        model.player_c_delta.weight.zero_()
        # W = [[2, 1], [0, 3]]
        model.W_synergy.copy_(torch.tensor([[2.0, 1.0], [0.0, 3.0]]))

    # Roster with only players 1 and 2
    player_ids = torch.tensor([[1, 2, 0, 0, 0, 0]], dtype=torch.long)
    format_ids = torch.tensor([0], dtype=torch.long)

    c_eff = model.get_effective_communication(player_ids, format_ids)
    mask = player_ids != 0
    s_mat = model.compute_team_synergy_matrix(c_eff, mask)
    s_team = model.compute_team_synergy(c_eff, mask)

    # c_1 = [1, 0], c_2 = [0, 1]
    # S_12 = c_1^T * W * c_2 = [1, 0] * [[2, 1], [0, 3]] * [0, 1]^T = [2, 1] * [0, 1]^T = 1.0
    assert torch.isclose(s_mat[0, 0, 1], torch.tensor(1.0))
    # Total synergy for team {1, 2} is S_12 = 1.0
    assert torch.isclose(s_team[0, 0], torch.tensor(1.0))


def test_gradient_flow_and_regularization() -> None:
    model = DualEmbeddingChGKModel(num_players=20, num_editors=10, dim_k=4, dim_c=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    player_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    editor_ids = torch.tensor([[2]], dtype=torch.long)
    format_ids = torch.tensor([1], dtype=torch.long)
    target = torch.tensor([[1.0]])

    optimizer.zero_grad()
    out = model(player_ids=player_ids, editor_ids=editor_ids, format_ids=format_ids)
    loss = criterion(out["logits"], target) + 1e-4 * model.compute_l2_regularization()
    loss.backward()

    # Verify gradients are non-zero for active parameters
    assert model.player_k_emb.weight.grad is not None
    assert model.player_c_base.weight.grad is not None
    assert model.editor_emb.weight.grad is not None
    assert model.editor_b.weight.grad is not None
    assert model.W_synergy.grad is not None

    optimizer.step()
