if __name__ == "__main__":
    from petsc4py import PETSc
    fwk = PETSc.Fwk()
    fwk.create()
    fwk.registerDependence("${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents:TestIIB",
                           "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents:TestIB")
    fwk.registerComponent( "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents:TestIC")
    fwk.registerDependence("${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIIA",
                           "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIA")
    fwk.registerComponent("${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIIC")
    fwk.configure(fwk, 1)


