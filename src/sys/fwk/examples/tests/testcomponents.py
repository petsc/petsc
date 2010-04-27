from petsc4py import PETSc

def PetscFwkConfigureTestIIIA(fwk, state, component):
    if state == 0:
        #assert component is None
        component = PETSc.Vec().create(fwk.comm)
        return component
    else:
        assert isinstance(component, PETSc.Vec)
        return None
