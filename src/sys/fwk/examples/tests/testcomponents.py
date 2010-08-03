from petsc4py import PETSc

def TestIIIA(fwk, conf, component):
    if component is None:
        component = PETSc.Vec().create(fwk.comm)
        print "Created a new component TestIIIA"
    else:
        assert isinstance(component, PETSc.Vec)
    print "Using configuration: " + str(conf)
    return component
