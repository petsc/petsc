import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc


    
if __name__ == "__main__":
    fwk = PETSc.Fwk().create()
    fwk.setName("ex1")
    fwk.registerComponentURL("Electrolyte", "./electrolyte.py:Electrolyte")
    fwk.registerComponentURL("Viz",         "./viz.py:Viz")
    viz = fwk.getComponent("Viz")
    fwk.visit("init")
    for i in range(4):
        viz.call("viewRho")
    del viz
    del fwk
