import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.

    Args:
        param: parameters (scalar or np.array)
        grad: gradients (same shape as param)
        m: first moment (same shape)
        v: second moment (same shape)
        t: timestep (1-based)
        lr: learning rate
        beta1: decay for first moment
        beta2: decay for second moment
        eps: small constant for stability

    Returns:
        (param_new, m_new, v_new)
    """

    # ensure numpy arrays (handles scalar too)
    param = np.array(param, dtype=float)
    grad = np.array(grad, dtype=float)
    m = np.array(m, dtype=float)
    v = np.array(v, dtype=float)

    # Step 1: update first moment
    m = beta1 * m + (1 - beta1) * grad

    # Step 2: update second moment
    v = beta2 * v + (1 - beta2) * (grad ** 2)

    # Step 3: bias correction
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # Step 4: parameter update
    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)

    return param, m, v