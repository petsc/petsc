from petsc4py import PETSc

n     = 32         # number of nodes in each direction, excluding those at the boundary
delta = 1.0/(n+1)  # grid spacing
d     = 1          # number of degrees of freedom

N = (n,n,n)
M = (n,n,n,d)

# the size of the domain
L = [1., 3., 1.]


class Field:
    @staticmethod
    def init(f):
        da = PETSc.DA().create(comm = f.comm, dim=3,dof=d+1,sizes=N)
        rhoVec = PETSc.Vec()
        rhoVec.createSeq(comm=f.comm,size=N[0]*N[1]*N[2]*d)
        rho    = rhoVec.getArray().reshape((N[0],N[1],N[2],d))
        from math import sin as sin, pi as pi
        for i in range(N[0]):
            for j in range(N[1]):
                for k in range(N[2]):
                    for s in range(d):
                        rho[i,j,k,s] = sin(2*pi*i/N[0])*sin(2*pi*j/N[1])*sin(2*pi*k/N[2])
        f.compose("mesh", da)
        f.compose("rho",rhoVec)


