import torch

def test_directional_flipping():
    B, D, H, W = 1, 1, 4, 4
    L = H * W
    x = torch.arange(L, dtype=torch.float32).view(B, D, H, W)
    print("Input x[0,0]:\n", x[0, 0])

    # Manual Diagonal Scans
    def diagonal_zigzag_concat(x, flip_y=False):
        B, D, H, W = x.shape
        if flip_y:
            x = torch.flip(x, dims=[2])  # Flip vertically for ↙ and ↗
        x = x.view(B * D, H, W)
        diagonals = [x.diagonal(offset=o, dim1=1, dim2=2) for o in range(-H + 1, W)]
        out = torch.cat(diagonals, dim=1)
        return out.view(B, D, -1)

    # Diagonal directions
    x_d1 = diagonal_zigzag_concat(x)                      # ↘
    x_d2 = torch.flip(x_d1, dims=[-1])                    # ↖
    x_d3 = diagonal_zigzag_concat(x, flip_y=True)         # ↙
    x_d4 = torch.flip(x_d3, dims=[-1])                    # ↗

    # Cross scan
    x_flat = x.view(B, D, -1)                             # →
    x_transpose = torch.transpose(x, 2, 3).contiguous().view(B, D, -1)  # ↓
    x_left = torch.flip(x_flat, dims=[-1])                # ←
    x_up = torch.flip(x_transpose, dims=[-1])             # ↑

    # Simulate out_y (B, 8, D, L)
    out_y = torch.cat([
        x_d1, x_d2, x_d3, x_d4,
        x_flat, x_transpose, x_left, x_up
    ], dim=0).view(1, 8, 1, L)

    # ===== Reverse all backward directions to match their forward versions =====
    inv_diag = torch.flip(out_y[:, [1, 3]], dims=[-1]).view(B, 2, -1, L)
    inv_cross = torch.flip(out_y[:, [6, 7]], dims=[-1]).view(B, 2, -1, L)

    vertical = torch.transpose(out_y[:, 5].view(B, -1, W, H), 2, 3).contiguous().view(B, -1, L)
    inv_vertical = torch.transpose(inv_cross[:, 1].view(B, -1, W, H), 2, 3).contiguous().view(B, -1, L)

    # ==== Final Comparison Output ====
    print("\n↘ Diagonal Flattened:\n", out_y[0, 0, 0])
    print("↖ Flipped Back:\n", inv_diag[0, 0, :])

    print("\n↙ Anti-Diagonal Flattened:\n", out_y[0, 2, 0])
    print("↗ Flipped Back:\n", inv_diag[0, 1, :])

    print("\n→ Horizontal:\n", out_y[0, 4, 0])
    print("← Flipped Back:\n", inv_cross[0, 0, :])

    print("\n↓ Vertical:\n", vertical[0])
    print("↑ Flipped Back:\n", inv_vertical[0])

    # Optional: assert equality to validate correctness
    assert torch.allclose(out_y[0, 0, 0], inv_diag[0, 0, :]), "↘ vs flipped ↖ mismatch"
    assert torch.allclose(out_y[0, 2, 0], inv_diag[0, 1, :]), "↙ vs flipped ↗ mismatch"
    assert torch.allclose(out_y[0, 4, 0], inv_cross[0, 0, :]), "→ vs flipped ← mismatch"
    assert torch.allclose(vertical[0], inv_vertical[0]), "↓ vs flipped ↑ mismatch"

    print("\n✅ All directions flipped correctly!")

test_directional_flipping()
