# file: bratu2dnpy.py

def bratu2d(alpha, x, f):
    # get 'exp' from numpy
    from numpy import exp
    # setup 5-points stencil
    u  = x[1:-1, 1:-1] # center
    uN = x[1:-1,  :-2] # north
    uS = x[1:-1, 2:  ] # south
    uW = x[ :-2, 1:-1] # west
    uE = x[2:,   1:-1] # east
    # compute nonlinear function
    nx, ny = x.shape
    hx = 1.0/(nx-1) # x grid spacing
    hy = 1.0/(ny-1) # y grid spacing
    f[:,:] = x
    f[1:-1, 1:-1] = \
         (2*u - uE - uW) * (hy/hx) \
       + (2*u - uN - uS) * (hx/hy) \
       - alpha * exp(u)  * (hx*hy)
