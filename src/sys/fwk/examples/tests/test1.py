import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

fwk = PETSc.Fwk().create()
fwk.registerComponent("TestIA", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIA")
fwk.registerComponent("TestIB", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIB")
fwk.registerComponent("TestIIA", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIIA")
fwk.registerComponent("TestIIB", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIIB")
fwk.registerComponent("TestIIIA", "./testcomponents.py:TestIIIA")
fwk.registerDependence("TestIIIA", "TestIA")
fwk.registerDependence("TestIIIA", "TestIIA")
fwk.view()
fwk.configure("test")

del fwk
