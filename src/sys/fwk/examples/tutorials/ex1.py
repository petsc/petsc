import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

n     = 32         # number of nodes in each direction, excluding those at the boundary
delta = 1.0/(n+1)  # grid spacing
d     = 2          # number of species

N = (n,n,n)
M = (n,n,n,d+1)

# the size of the domain
L = [1., 3., 1.]

# bulk Gamma
GAMMA = 1.0

def Electrolyte(fwk, key, stage, e):
    if e is None:
        # Create a Fwk to serve as the component
        e = PETSc.Fwk().create(fwk.comm)
        return e
    if stage == "init":
        da = PETSc.DA().create(dim=3,dof=d+1,sizes=N)
        rhoVec = PETSc.Vec()
        rhoVec.createSeq(comm=da.comm,size=N[0]*N[1]*N[2]*d)
        rho    = rhoVec.getArray()
        rho    = rho.reshape((N[0],N[1],N[2],d))
        from math import sin as sin, pi as pi
        for i in range(N[0]):
            for j in range(N[1]):
                for k in range(N[2]):
                    for s in range(d):
                        rho[i,j,k,s] = sin(2*pi*i/N[0])*sin(2*pi*j/N[1])*sin(2*pi*k/N[2])
        e.compose("mesh", da)
        e.compose("rho",rhoVec)
    return e
    
if __name__ == "__main__":
    fwk = PETSc.Fwk().create()
    fwk.registerComponent("Electrolyte", "./ex1.py:Electrolyte")
    fwk.registerComponent("Viz",         "./viz.py:Viz")
    fwk.configure("init")
    for i in range(4):
        fwk.configure("viewRho")
    del fwk
