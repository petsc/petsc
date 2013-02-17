import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc
import Bratu3D as Bratu3D

OptDB = PETSc.Options()

N = OptDB.getInt('N', 16)
lambda_ = OptDB.getReal('lambda', 6.0)
do_plot = OptDB.getBool('plot', False)

da = PETSc.DMDA().create([N, N, N], stencil_width=1)
#app = App(da, lambda_)

snes = PETSc.SNES().create()
F = da.createGlobalVec()
snes.setFunction(Bratu3D.formFunction, F,
                 args=(da, lambda_))
J = da.createMat()
snes.setJacobian(Bratu3D.formJacobian, J,
                 args=(da, lambda_))

snes.setFromOptions()

X = da.createGlobalVec()
Bratu3D.formInitGuess(X, da, lambda_)
snes.solve(None, X)

U = da.createNaturalVec()
da.globalToNatural(X, U)


def plot(da, U):
    scatter, U0 = PETSc.Scatter.toZero(U)
    scatter.scatter(U, U0, False, PETSc.Scatter.Mode.FORWARD)
    rank = PETSc.COMM_WORLD.getRank()
    if rank == 0:
        solution = U0[...].reshape(da.sizes, order='f').copy()
        try:
            from matplotlib import pyplot
            pyplot.contourf(solution[:, :, N//2])
            pyplot.axis('equal')
            pyplot.show()
        except:
            pass
    PETSc.COMM_WORLD.barrier()
    scatter.destroy()
    U0.destroy()


if do_plot: plot(da, U)


U.destroy()
X.destroy()
F.destroy()
J.destroy()
da.destroy()
snes.destroy()
