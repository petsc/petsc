import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc
import Bratu2D as Bratu2D

class App(object):

    def __init__(self, da, lambda_):
        assert da.getDim() == 2
        self.da = da
        self.lambda_ = lambda_

    def formInitGuess(self, snes, X):
        X.zeroEntries() # just in case
        da = self.da.fortran
        vec_X = X.fortran
        ierr = Bratu2D.FormInitGuess(da, vec_X, self.lambda_)
        if ierr: raise PETSc.Error(ierr)

    def formFunction(self, snes, X, F):
        F.zeroEntries() # just in case
        da = self.da.fortran
        vec_X = X.fortran
        vec_F = F.fortran
        ierr = Bratu2D.FormFunction(da, vec_X, vec_F, self.lambda_)
        if ierr: raise PETSc.Error(ierr)

    def formJacobian(self, snes, X, J, P):
        P.zeroEntries() # just in case
        da = self.da.fortran
        vec_X = X.fortran
        mat_P = P.fortran
        ierr = Bratu2D.FormJacobian(da, vec_X, mat_P, self.lambda_)
        if ierr: raise PETSc.Error(ierr)
        if J != P: J.assemble() # matrix-free operator
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN


OptDB = PETSc.Options()

N = OptDB.getInt('N', 16)
lambda_ = OptDB.getReal('lambda', 6.0)
do_plot = OptDB.getBool('plot', False)

da = PETSc.DA().create([N, N], stencil_width=1)
app = App(da, lambda_)

snes = PETSc.SNES().create()
F = da.createGlobalVec()
snes.setFunction(app.formFunction, F)
J = da.createMat()
snes.setJacobian(app.formJacobian, J)

snes.setFromOptions()

X = da.createGlobalVec()
app.formInitGuess(snes, X)
snes.solve(None, X)

U = da.createNaturalVec()
da.globalToNatural(X, U)

def plot(da, U):
    comm = da.getComm()
    scatter, U0 = PETSc.Scatter.toZero(U)
    scatter.scatter(U, U0, False, PETSc.Scatter.Mode.FORWARD)
    rank = comm.getRank()
    if rank == 0:
        solution = U0[...]
        solution = solution.reshape(da.sizes, order='f').copy()
        try:
            from matplotlib import pyplot
            pyplot.contourf(solution)
            pyplot.axis('equal')
            pyplot.show()
        except:
            pass
    comm.barrier()
    scatter.destroy()
    U0.destroy()

if do_plot: plot(da, U)


U.destroy()
X.destroy()
F.destroy()
J.destroy()
da.destroy()
snes.destroy()
