from petsc4py import PETSc

def TestIIIA(fwk, key, conf, component):
    if component is None:
        component = PETSc.Vec().create(fwk.comm)
        print "Created a new component " + str(key)
    else:
        assert isinstance(component, PETSc.Vec)
    print "TestIIIA: using configuration: " + str(conf)
    return component
