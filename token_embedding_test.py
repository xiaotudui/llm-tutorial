import torch

def rotary_position_embedding(max_seq_len, dim):
    # Calculate the angle rates based on dimension indices.
    angle_rates = 1 / torch.pow(10000, torch.arange(0, dim, 2).float() / dim)
    # Calculate the angles for each position for half of the dimensions (sine and cosine)
    angles = (torch.arange(max_seq_len).unsqueeze(1) * angle_rates.unsqueeze(0))
    # Cosines and sines of the angles to get the RoPE for each position
    position_encodings = torch.stack((angles.cos(), angles.sin()), dim=2).flatten(1)
    return position_encodings

def apply_rope_embeddings(embeddings, position_encodings):
    # Split the position encodings into cosines and sines
    cos_enc, sin_enc = position_encodings[..., 0::2], position_encodings[..., 1::2]
    # Apply the rotations
    embeddings[..., 0::2] = embeddings[..., 0::2] * cos_enc - embeddings[..., 1::2] * sin_enc
    embeddings[..., 1::2] = embeddings[..., 1::2] * cos_enc + embeddings[..., 0::2] * sin_enc
    return embeddings

batch_size, max_seq_len, dim = 1, 128, 512  # Dimensions for batch size, sequence length, and embedding dimension

# Initialize random embeddings simulating a batch of token embeddings
token_embeddings = torch.randn(batch_size, max_seq_len, dim)

# Generate the position encodings for the sequence
position_encodings = rotary_position_embedding(max_seq_len, dim)

# Apply the RoPE to the token embeddings
rotated_token_embeddings = apply_rope_embeddings(token_embeddings, position_encodings)