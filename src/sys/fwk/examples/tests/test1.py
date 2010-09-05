import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

fwk = PETSc.Fwk().create()
fwk.registerComponentURL("IA", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIA")
fwk.registerComponentURL("IB", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIB")
fwk.registerComponentURL("IIA", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIIA")
fwk.registerComponentURL("IIB", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIIB")
fwk.registerComponentURL("IIIA", "./testcomponents.py:TestIIIA")
fwk.registerDependence("IIIA", "IA")
fwk.registerDependence("IIIA", "IIA")
fwk.view()
fwk.visit("test")

del fwk
