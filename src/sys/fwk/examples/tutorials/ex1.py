import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc


    
if __name__ == "__main__":
    fwk = PETSc.Fwk().create()
    fwk.setName("ex1")
    fwk.registerComponent("Electrolyte", "./electrolyte.py:Electrolyte")
    fwk.registerComponent("Viz",         "./viz.py:Viz")
    viz = fwk.getComponent("Viz")
    print "ex1 framework prior to an 'init' visit:"
    fwk.visit("init")
    print "ex1 framework after an 'init' visit:"
    fwk.view()
    for i in range(4):
        viz.call("viewRho")
    del viz
    del fwk
