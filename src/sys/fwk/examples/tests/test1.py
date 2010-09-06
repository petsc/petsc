import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

fwk = PETSc.Fwk().create()
fwk.registerComponentURL("TestIA", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIA")
fwk.registerComponentURL("TestIB", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIB")
fwk.registerComponentURL("TestIIA", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIIA")
fwk.registerComponentURL("TestIIB", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIIB")
fwk.registerComponentURL("TestIIIA", "./testcomponents.py:TestIIIA")
fwk.registerDependence("TestIIIA", "TestIA")
fwk.registerDependence("TestIIIA", "TestIIA")
print "Viewing top-level framework:"
fwk.view()
message = "initialize"
print "Visiting with message '" + str(message) + "'"
fwk.visit(message)

print "Viewing top-level framework:"
fwk.view()
message = "configure"
print "Visiting with message '" + str(message) + "'"
fwk.visit(message)

del fwk
