#!/usr/bin/python
import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc


    
if __name__ == "__main__":
    field = PETSc.Shell().create().setURL("./field.py:Field")
    viz   = PETSc.Shell().create().setURL("./viz1.py:Viz")
    field.call("init")
    viz.compose("mesh",field.query("mesh"))
    viz.compose("rho", field.query("rho"))
    viz.call("viewRho")
    del viz
    del field
