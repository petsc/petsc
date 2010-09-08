import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc


if __name__ == "__main__":
    fwk = PETSc.Fwk().create()
    fwk.setName("ex1")
    fwk.registerComponent("Electrolyte",  "./electrolyte.py:Electrolyte")
    fwk.registerComponent("Viz",          "./viz.py:Viz")
    #fwk.registerComponent("ScreeningAvg", "./avg.py:ScreeningAvg")
    fwk.visit("init")
    viz = fwk.getComponent("Viz")
    #viz.call("viewRhoGamma")
    viz.call("viewRho")
    rhoGamma = fwk.getComponent("Electrolyte").query("rhoGamma")
    rho      = fwk.getComponent("Electrolyte").query("rho")
    #A        = fwk.getComponent("ScreeningAvg")
    #A.mult(rhoGamma,rho)
    

    
