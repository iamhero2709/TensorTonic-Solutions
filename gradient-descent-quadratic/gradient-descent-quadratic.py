def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Gradient descenet in 1d 
    x=x0
    for i in range (steps):
        # gradient
        grad=2*a*x+b
        # update gradient
        x=x-lr*grad
    return x 
    pass