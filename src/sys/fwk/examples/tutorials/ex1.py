import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

def DensityField(fwk, key, conf, c):
    if c is None:
        # Create a Fwk to serve as the component
        c = PETSc.Fwk().create(fwk.comm)
        return c
    if conf == "init":
        N = 32
        d = 2
        da = PETSc.DA().create(dim=3,dof=d,sizes=(N,N,N))
        vec = da.createNaturalVector()
        v = vec.getArray()
        v = v.reshape((N,N,N,d))
        from math import sin as sin, pi as pi
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(d):
                        v[i,j,k,l] = sin(2*pi*i/N)*sin(2*pi*j/N)*sin(2*pi*k/N)
        c.compose("mesh", da)
        c.compose("density",vec)
    return c
    
if __name__ == "__main__":
    fwk = PETSc.Fwk().create()
    fwk.registerComponent("DensityField", "./ex1.py:DensityField")
    fwk.registerComponent("Viz", "./ex1_viz.py:Viz")
    #fwk.view()
    fwk.configure("init")

    for i in range(10):
        fwk.configure("run")

    del fwk
