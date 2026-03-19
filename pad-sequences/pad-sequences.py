import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L)
    """
    
    N = len(seqs)
    
    #  find target length
    if max_len is not None:
        L = max_len
    else:
        L = max(len(seq) for seq in seqs) if seqs else 0

    #  create output array (filled with pad_value)
    result = np.full((N, L), pad_value)

    #  fill values
    for i, seq in enumerate(seqs):
        length = min(len(seq), L)  # truncate if needed
        result[i, :length] = seq[:length]

    return result