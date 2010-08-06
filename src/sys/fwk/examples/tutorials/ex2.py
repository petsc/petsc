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

# species' radius
RADIUS = 1.0

def Electrolyte(fwk, key, stage, e):
    if e is None:
        # Create a Fwk to serve as the Electrolyte component
        e = PETSc.Fwk().create(fwk.comm)
        #fwk.registerDependence(key,"ScreeningAvg")
        return e
    if stage == "init":
        da = PETSc.DA().create(dim=3,dof=d+1,sizes=N)
        #
        hVec  = PETSc.Vec().createSeq(3)
        h     = hVec.getArray()
        h[:]  = [delta, delta, delta]
        #
        radiiVec  = PETSc.Vec().createSeq(d)
        radii     = radiiVec.getArray()
        radii[:]  = [RADIUS for i in range(d)]
        e.compose("mesh", da)
        e.compose("h",    hVec)
        e.compose("radii",radiiVec)
        #
        rhoVec = PETSc.Vec().createSeq(comm=da.comm, size=N[0]*N[1]*N[2]*d)
        e.compose("rho", rhoVec)
        rhoGammaVec = PETSc.Vec().createSeq(comm=da.comm, size=N[0]*N[1]*N[2]*(d+1))
        e.compose("rhoGamma", rhoGammaVec)
    if stage == "viewRho":
        #
        da = e.query("mesh")
        #
        rho    = e.query("rho").getArray()
        rho    = rho.reshape((N[0],N[1],N[2],d))
        from math import sin as sin, pi as pi
        for i in range(N[0]):
            for j in range(N[1]):
                for k in range(N[2]):
                    for s in range(d):
                        rho[i,j,k,s] = sin(2*pi*i/N[0])*sin(2*pi*j/N[1])*sin(2*pi*k/N[2])
    if stage == "viewRhoGamma":
        #
        da = e.query("mesh")
        #
        rhoGamma    = e.query("rhoGamma").getArray()
        rhoGamma    = rhoGamma.reshape((N[0],N[1],N[2],d+1))
        from math import sin as sin, pi as pi
        for i in range(N[0]):
            for j in range(N[1]):
                for k in range(N[2]):
                    for s in range(d):
                        rhoGamma[i,j,k,s] = sin(2*pi*i/N[0])*sin(2*pi*j/N[1])*sin(2*pi*k/N[2])
                    rhoGamma[i,j,k,d] = GAMMA
    return e


if __name__ == "__main__":
    fwk = PETSc.Fwk().create()
    fwk.registerComponent("Electrolyte",  "./ex2.py:Electrolyte")
    fwk.registerComponent("Viz",          "./viz.py:Viz")
    #fwk.registerComponent("ScreeningAvg", "./avg.py:ScreeningAvg")
    fwk.configure("init")
    fwk.configure("viewRhoGamma")
    rhoGamma = fwk.getComponent("Electrolyte").query("rhoGamma")
    rho      = fwk.getComponent("Electrolyte").query("rho")
    #A        = fwk.getComponent("ScreeningAvg")
    #A.mult(rhoGamma,rho)
    fwk.configure("viewRho")
    

    
