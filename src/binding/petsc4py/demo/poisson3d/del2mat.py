# file: del2mat.py

from numpy import zeros
from del2lib import del2apply

class Del2Mat:

    def __init__(self, n=1):
        self.N = (n, n, n)
        self.F = zeros([n+2]*3, order='f')

    def create(self, A):
        N = self.N
        mat_size = A.getSize()
        grid_eqs = N[0]*N[1]*N[2]
        assert mat_size[0] == grid_eqs
        assert mat_size[1] == grid_eqs

    def mult(self, A, x, y):
        "y <- A * x"
        N, F = self.N, self.F
        # get 3D arrays from vectos
        xx = x.getArray(readonly=1).reshape(N, order='f')
        yy = y.getArray(readonly=0).reshape(N, order='f')
        # call Fortran subroutine
        del2apply(F, xx, yy)

    def multTranspose(self, A, x, y):
        "y <- A' * x"
        self.mult(x, y)

    def getDiagonal(self, A, D):
        "D[i] <- A[i,i]"
        D[...] = 6.0
