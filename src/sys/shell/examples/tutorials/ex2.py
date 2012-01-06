#!/usr/bin/python
import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc


if __name__ == "__main__":
    shell = PETSc.Shell().create()
    shell.setName("ex2")
    shell.registerComponent("Field",  url = "./field.py:Field")
    shell.registerComponent("Viz",    url = "./viz2.py:Viz")
    shell.registerDependence("Field", "Viz")
    shell.visit("init")
    viz = shell.getComponent("Viz")
    viz.call("viewRho")

