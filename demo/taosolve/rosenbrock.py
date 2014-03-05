"""
This example demonstrates the use of TAO for Python to solve an
unconstrained minimization problem on a single processor.

We minimize the extended Rosenbrock function::

   sum_{i=0}^{n/2-1} ( alpha*(x_{2i+1}-x_{2i}^2)^2 + (1-x_{2i})^2 )
"""

try: range = xrange
except NameError: pass

# the two lines below are only
# needed to build options database
# from command line arguments
import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc


class AppCtx(object):

    """
    Extended Rosenbrock function.
    """

    def __init__(self, n=2, alpha=99.0):
        self.size  = int(n)
        self.alpha = float(alpha)

    def formObjective(self, tao, x):
        #print 'AppCtx.formObjective()'
        alpha = self.alpha
        nn = self.size // 2
        ff = 0.0
        for i in range(nn):
            t1 = x[2*i+1] - x[2*i] * x[2*i]
            t2 = 1 - x[2*i];
            ff += alpha*t1*t1 + t2*t2;
        return ff

    def formGradient(self, tao, x, G):
        #print 'AppCtx.formGradient()'
        alpha = self.alpha
        nn = self.size // 2
        G.zeroEntries()
        for i in range(nn):
            t1 = x[2*i+1] - x[2*i] * x[2*i]
            t2 = 1 - x[2*i];
            G[2*i]   = -4*alpha*t1*x[2*i] - 2*t2;
            G[2*i+1] = 2*alpha*t1;

    def formObjGrad(self, tao, x, G):
        #print 'AppCtx.formObjGrad()'
        alpha = self.alpha
        nn = self.size // 2
        ff = 0.0
        G.zeroEntries()
        for i in range(nn):
            t1 = x[2*i+1] - x[2*i] * x[2*i]
            t2 = 1 - x[2*i];
            ff += alpha*t1*t1 + t2*t2;
            G[2*i]   = -4*alpha*t1*x[2*i] - 2*t2;
            G[2*i+1] = 2*alpha*t1;
        return ff

    def formHessian(self, tao, x, H, HP):
        #print 'AppCtx.formHessian()'
        alpha = self.alpha
        nn = self.size // 2
        idx = [0, 0]
        v = [[0.0, 0.0],
             [0.0, 0.0]]
        H.zeroEntries()
        for i in range(nn):
            v[1][1] = 2*alpha
            v[0][0] = -4*alpha*(x[2*i+1]-3*x[2*i]*x[2*i]) + 2
            v[1][0] = v[0][1] = -4.0*alpha*x[2*i];
            idx[0] = 2*i
            idx[1] = 2*i+1
            H[idx,idx] = v
        H.assemble()

# access PETSc options database
OptDB = PETSc.Options()


# create user application context
# and configure user parameters
user = AppCtx()
user.size  = OptDB.getInt (    'n', user.size)
user.alpha = OptDB.getReal('alpha', user.alpha)

# create solution vector
x = PETSc.Vec().create(PETSc.COMM_SELF)
x.setSizes(user.size)
x.setFromOptions()

# create Hessian matrix
H = PETSc.Mat().create(PETSc.COMM_SELF)
H.setSizes([user.size, user.size])
H.setFromOptions()
H.setOption(PETSc.Mat.Option.SYMMETRIC, True)
H.setUp()

# pass the following to command line:
#  $ ... -methods nm,lmvm,nls,ntr,cg,blmvm,tron
# to try many methods
methods = OptDB.getString('methods', '')
methods = methods.split(',')
for meth in methods:
    # create TAO Solver
    tao = PETSc.TAO().create(PETSc.COMM_SELF)
    if meth: tao.setType(meth)
    tao.setFromOptions()
    # solve the problem
    tao.setObjectiveGradient(user.formObjGrad)
    tao.setObjective(user.formObjective)
    tao.setGradient(user.formGradient)
    tao.setHessian(user.formHessian, H)
    #app.getKSP().getPC().setFromOptions()
    x.set(0) # zero initial guess
    #tao.setInitial(x)
    tao.solve(x)
    tao.destroy()

## # this is just for testing
## x     = app.getSolution()
## G     = app.getGradient()
## H, HP = app.getHessian()

## f = tao.computeObjective(x)
## tao.computeGradient(x, G)
## f = tao.computeObjectiveGradient(x, G)
## tao.computeHessian(x, H, HP)
