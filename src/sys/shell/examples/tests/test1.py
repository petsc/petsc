#!/usr/bin/python
import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

shell = PETSc.Shell().create()
shell.registerComponent("TestIA", url="${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIA")
shell.registerComponent("TestIB", url="${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIB")
shell.registerComponent("TestIIA", url="${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIIA")
shell.registerComponent("TestIIB", url="${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIIB")
shell.registerComponent("TestIIIA", url="./testcomponentsIII.py:TestIIIA")

sys.stdout.write("Registering dependence TestIIIA --> TestIA\n")
shell.registerDependence("TestIIIA", "TestIA")
sys.stdout.write("Registering dependence TestIIIA --> TestIIA\n")
shell.registerDependence("TestIIIA", "TestIIA")
sys.stdout.flush()



sys.stdout.write("Viewing top-level shell:\n")
sys.stdout.flush()
shell.view()
message = "initialize"
sys.stdout.write("Visiting with message '" + str(message) + "'\n")
sys.stdout.flush()
shell.visit(message)

sys.stdout.write("Registering dependence TestIIB --> TestIB\n")
sys.stdout.flush()
shell.registerDependence("TestIIB", "TestIB")

sys.stdout.write("Viewing top-level shell:\n")
sys.stdout.flush()
shell.view()
message = "configure"
sys.stdout.write("Visiting with message '" + str(message) + "'\n")
sys.stdout.flush()
shell.visit(message)

del shell
