# Solves Heat equation on a periodic domain, using raw VecScatter
from __future__ import division
import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc
from mpi4py import MPI
import numpy

class Heat(object):
    def __init__(self,comm,N):
        self.comm = comm
        self.N = N              # global problem size
        self.h = 1/N            # grid spacing on unit interval
        self.n = N // comm.size + int(comm.rank < (N % comm.size)) # owned part of global problem
        self.start = comm.exscan(self.n)
        if comm.rank == 0: self.start = 0
        gindices = numpy.arange(self.start-1, self.start+self.n+1, dtype=PETSc.IntType) % N # periodic
        self.mat = PETSc.Mat().create(comm=comm)
        size = (self.n, self.N) # local and global sizes
        self.mat.setSizes((size,size))
        self.mat.setFromOptions()
        self.mat.setPreallocationNNZ((3,1)) # Conservative preallocation for 3 "local" columns and one non-local

        # Allow matrix insertion using local indices [0:n+2]
        lgmap = PETSc.LGMap().create(list(gindices), comm=comm)
        self.mat.setLGMap(lgmap, lgmap)

        # Global and local vectors
        self.gvec = self.mat.createVecRight()
        self.lvec = PETSc.Vec().create(comm=PETSc.COMM_SELF)
        self.lvec.setSizes(self.n+2)
        self.lvec.setUp()
        # Configure scatter from global to local
        isg = PETSc.IS().createGeneral(list(gindices), comm=comm)
        self.g2l = PETSc.Scatter().create(self.gvec, isg, self.lvec, None)

        self.tozero, self.zvec = PETSc.Scatter.toZero(self.gvec)
        self.history = []

        if False:                # Print some diagnostics
            print('[%d] local size %d, global size %d, starting offset %d' % (comm.rank, self.n, self.N, self.start))
            self.gvec.setArray(numpy.arange(self.start,self.start+self.n))
            self.gvec.view()
            self.g2l.scatter(self.gvec, self.lvec, PETSc.InsertMode.INSERT)
            for rank in range(comm.size):
                if rank == comm.rank:
                    print('Contents of local Vec on rank %d' % rank)
                    self.lvec.view()
                comm.barrier()
    def evalSolution(self, t, x):
        assert t == 0.0, "only for t=0.0"
        coord = numpy.arange(self.start, self.start+self.n) / self.N
        x.setArray((numpy.abs(coord-0.5) < 0.1) * 1.0)
    def evalFunction(self, ts, t, x, xdot, f):
        self.g2l.scatter(x, self.lvec, PETSc.InsertMode.INSERT) # lvec is a work vector
        h = self.h
        with self.lvec as u, xdot as udot:
            f.setArray(udot*h + 2*u[1:-1]/h - u[:-2]/h - u[2:]/h) # Scale equation by volume element
    def evalJacobian(self, ts, t, x, xdot, a, A, B):
        h = self.h
        for i in range(self.n):
            lidx = i + 1
            gidx = self.start + i
            B.setValuesLocal([lidx], [lidx-1,lidx,lidx+1], [-1/h, a*h+2/h, -1/h])
        B.assemble()
        if A != B: A.assemble() # If operator is different from preconditioning matrix
        return True # same nonzero pattern
    def monitor(self, ts, i, t, x):
        if self.history:
            lasti, lastt, lastx = self.history[-1]
            if i < lasti + 4 or t < lastt + 1e-4: return
        self.tozero.scatter(x, self.zvec, PETSc.InsertMode.INSERT)
        xx = self.zvec[:].tolist()
        self.history.append((i, t, xx))
    def plotHistory(self):
        try:
            from matplotlib import pylab, rcParams
        except ImportError:
            print("matplotlib not available")
            raise SystemExit
        rcParams.update({'text.usetex':True, 'figure.figsize':(10,6)})
        #rc('figure', figsize=(600,400))
        pylab.title('Heat: TS \\texttt{%s}' % ts.getType())
        x = numpy.arange(self.N) / self.N
        for i,t,u in self.history:
            pylab.plot(x, u, label='step=%d t=%8.2g'%(i,t))
        pylab.xlabel('$x$')
        pylab.ylabel('$u$')
        pylab.legend(loc='upper right')
        pylab.savefig('heat-history.png')
        #pylab.show()

OptDB = PETSc.Options()
ode = Heat(MPI.COMM_WORLD, OptDB.getInt('n',100))

x = ode.gvec.duplicate()
f = ode.gvec.duplicate()

ts = PETSc.TS().create(comm=ode.comm)
ts.setType(ts.Type.ROSW)        # Rosenbrock-W. ARKIMEX is a nonlinearly implicit alternative.

ts.setIFunction(ode.evalFunction, ode.gvec)
ts.setIJacobian(ode.evalJacobian, ode.mat)

ts.setMonitor(ode.monitor)

ts.setTime(0.0)
ts.setTimeStep(ode.h**2)
ts.setMaxTime(1)
ts.setMaxSteps(100)
ts.setExactFinalTime(PETSc.TS.ExactFinalTime.INTERPOLATE)
ts.setMaxSNESFailures(-1)       # allow an unlimited number of failures (step will be rejected and retried)

snes = ts.getSNES()             # Nonlinear solver
snes.setTolerances(max_it=10)   # Stop nonlinear solve after 10 iterations (TS will retry with shorter step)
ksp = snes.getKSP()             # Linear solver
ksp.setType(ksp.Type.CG)        # Conjugate gradients
pc = ksp.getPC()                # Preconditioner
if False:                       # Configure algebraic multigrid, could use run-time options instead
    pc.setType(pc.Type.GAMG)    # PETSc's native AMG implementation, mostly based on smoothed aggregation
    OptDB['mg_coarse_pc_type'] = 'svd' # more specific multigrid options
    OptDB['mg_levels_pc_type'] = 'sor'

ts.setFromOptions()             # Apply run-time options, e.g. -ts_adapt_monitor -ts_type arkimex -snes_converged_reason
ode.evalSolution(0.0, x)
ts.solve(x)
if ode.comm.rank == 0:
    print('steps %d (%d rejected, %d SNES fails), nonlinear its %d, linear its %d'
          % (ts.getStepNumber(), ts.getStepRejections(), ts.getSNESFailures(),
             ts.getSNESIterations(), ts.getKSPIterations()))

if OptDB.getBool('plot_history', True) and ode.comm.rank == 0:
    ode.plotHistory()
