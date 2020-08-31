def cg(A, b, x, imax=50, eps=1e-6):
    """
    A, b, x  : matrix, rhs, solution
    imax     : maximum allowed iterations
    eps      : tolerance for convergence
    """
    # allocate work vectors
    r = b.duplicate()
    d = b.duplicate()
    q = b.duplicate()
    # initialization
    i = 0
    A.mult(x, r)
    r.aypx(-1, b)
    r.copy(d)
    delta_0 = r.dot(r)
    delta = delta_0
    # enter iteration loop
    while i < imax and \
          delta > delta_0 * eps**2:
        A.mult(d, q)
        alpha = delta / d.dot(q)
        x.axpy(+alpha, d)
        r.axpy(-alpha, q)
        delta_old = delta
        delta = r.dot(r)
        beta = delta / delta_old
        d.aypx(beta, r)
        i = i + 1
    return i, delta**0.5
