import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

fwk = PETSc.Fwk().create()
fwk.registerComponent("IA", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIA")
fwk.registerComponent("IB", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIB")
fwk.registerComponent("IIA", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIIA")
fwk.registerComponent("IIB", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIIB")
fwk.registerComponent("IIIA", "./testcomponents.py:TestIIIA")
fwk.registerDependence("IIIA", "IA")
fwk.registerDependence("IIIA", "IIA")
fwk.view()
fwk.configure("test")

del fwk
