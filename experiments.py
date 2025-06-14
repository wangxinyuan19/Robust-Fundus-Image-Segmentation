import torch

class SS2D:
    @staticmethod
    def diagonal_zigzag_concat(x):
        B, D, H, W = x.shape
        x = x.view(B * D, H, W)

        diagonals = [x.diagonal(offset=o, dim1=1, dim2=2) for o in range(-H + 1, W)]
        out = torch.cat(diagonals, dim=1)  # (B*D, L)
        return out.view(B, D, -1)
    
    @staticmethod
    def reconstruct_from_diagonal_zigzag(x_diag, N, direction="↘"):
        """
        Vectorized diagonal reconstruction from zigzag sequence.
        Assumes square input (N x N).
        x_diag: shape (B, D, N*N)
        direction: "↘", "↙", "↗", "↖"
        """
        B, D, L = x_diag.shape
        assert L == N * N, f"Expected flattened size {N*N}, got {L}"

        # Flip helpers
        def flip(x, direction):
            if direction == "↙":
                return torch.flip(x, dims=[-1])
            elif direction == "↗":
                return torch.flip(x, dims=[-2])
            elif direction == "↖":
                return torch.flip(x, dims=[-1, -2])
            else:
                return x

        # Total number of diagonals
        num_diags = 2 * N - 1
        device = x_diag.device

        # Precompute all (row, col) indices for each diagonal
        coords = []
        for offset in range(-N + 1, N):
            if offset >= 0:
                r = torch.arange(N - offset)
                c = r + offset
            else:
                c = torch.arange(N + offset)
                r = c - offset
            coords.append(torch.stack([r, c], dim=1))  # shape: (len, 2)

        coords = torch.cat(coords, dim=0)  # shape: (N*N, 2)
        row_idx, col_idx = coords[:, 0], coords[:, 1]  # each of shape (N*N,)

        # Expand to full shape
        base = torch.zeros(B * D, N, N, device=device)
        flat = x_diag.view(B * D, -1)

        # Create flattened row/col indices for scatter_add_
        row_idx = row_idx.unsqueeze(0).expand(B * D, -1)
        col_idx = col_idx.unsqueeze(0).expand(B * D, -1)
        batch_idx = torch.arange(B * D, device=device).unsqueeze(1).expand_as(row_idx)

        # Use scatter_add to reconstruct
        recon = torch.zeros(B * D, N, N, device=device)
        recon.index_put_((batch_idx, row_idx, col_idx), flat, accumulate=True)

        recon = recon.view(B, D, N, N)
        return flip(recon, direction)





# Define a 4x4 image with values 1 to 16
x = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4)

# Create flipped versions
x_flip_h = torch.flip(x, dims=[3])     # horizontal flip (↙)
x_flip_v = torch.flip(x, dims=[2])     # vertical flip (↗)
x_flip_hv = torch.flip(x, dims=[2, 3]) # horizontal + vertical flip (↖)

# Apply diagonal scan
x_main_d = SS2D.diagonal_zigzag_concat(x)        # ↘
x_flip_h_d = SS2D.diagonal_zigzag_concat(x_flip_h)  # ↙
x_flip_v_d = SS2D.diagonal_zigzag_concat(x_flip_v)  # ↗
x_flip_hv_d = SS2D.diagonal_zigzag_concat(x_flip_hv) # ↖

x = torch.arange(1, 17).reshape(1, 1, 4, 4).float()
x_recon_main = SS2D.reconstruct_from_diagonal_zigzag(x_main_d, 4, "↘")
x_recon_h = SS2D.reconstruct_from_diagonal_zigzag(x_flip_h_d, 4, "↙")
x_recon_v = SS2D.reconstruct_from_diagonal_zigzag(x_flip_v_d, 4, "↗")
x_recon_hv = SS2D.reconstruct_from_diagonal_zigzag(x_flip_hv_d, 4, "↖")

#xs = torch.cat([torch.stack([x_main_d, x_flip_h_d, x_flip_v_d, x_flip_hv_d], dim=1)])
# Horizontal and vertical directions
x_hv = torch.stack([x.view(1, -1, 16), torch.transpose(x, dim0=2, dim1=3).contiguous().view(1, -1, 16)], dim=1).view(1, 2, -1, 16)
flipped_x_hv = torch.flip(x_hv, dims=[-1])       
xs = torch.cat([torch.stack([x_main_d, x_flip_h_d, x_flip_v_d, x_flip_hv_d], dim=1), x_hv, flipped_x_hv], dim=1)  # (B, 8, D, L)
out_y = xs
H=4
B=1
L=16
W=4
# Reconstruct the 4 diagonal scanning paths back to matrix
recon_main_d = SS2D.reconstruct_from_diagonal_zigzag(out_y[:, 0], H, "↘").view(B, -1, L)
recon_h_d = SS2D.reconstruct_from_diagonal_zigzag(out_y[:, 1], H, "↙").view(B, -1, L)
recon_v_d = SS2D.reconstruct_from_diagonal_zigzag(out_y[:, 2], H, "↗").view(B, -1, L)
recon_hv_d = SS2D.reconstruct_from_diagonal_zigzag(out_y[:, 3], H, "↖").view(B, -1, L)

# Reconstruct the 4 horizontal and vertical scanning path to matrix
flipped_back_directions = torch.flip(out_y[:, 6:8], dims=[-1]).view(B, 2, -1, L)
recon_down_direction1 = torch.transpose(out_y[:, 5].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
recon_down_direction2 = torch.transpose(flipped_back_directions[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

y1, y2, y3, y4, y5, y6, y7, y8 = recon_main_d, recon_h_d, recon_v_d, recon_hv_d, out_y[:, 4], flipped_back_directions[:, 0], recon_down_direction1, recon_down_direction2
print("Original:\n", x.view(4, 4))
'''print("\n↘ Main diagonal zigzag:\n", x_flip_hv_d.view(-1))
print("Reconstructed:\n", x_recon_main.view(1, -1, 16))
print("Reconstructed:\n", x_recon_h.view(1, -1, 16))
print("Reconstructed:\n", x_recon_v.view(4, 4))
print("Reconstructed:\n", x_recon_hv.view(4, 4))
print("Summed up: \n", xs)'''

print(y1+y2+y3+y4+y5+y6+y7+y8)
