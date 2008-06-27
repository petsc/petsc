
def help(args=None):
    import sys, petsc4py
    petsc4py.init(sys.argv + ['-help'])
    from petsc4py import PETSc
    ARGS = args or sys.argv[1:]
    COMM = PETSc.COMM_SELF
    #if 'options' in args:
    #    opt = PETSc.Options()
    #    opt.setFromOptions()
    if 'vec' in ARGS:
        vec = PETSc.Vec().create(comm=COMM)
        vec.setSizes(0)
        vec.setFromOptions()
        del vec
    #if 'vecscatter' in ARGS:
    #    i = PETSc.IS().createGeneral([0], comm=COMM)
    #    v = PETSc.Vec().create(comm=COMM)
    #    v.setSizes(1)
    #    v.setType(PETSc.Vec.Type.SEQ)
    #    scatter = PETSc.Scatter().create(v,i,v,i)
    if 'mat' in ARGS:
        mat = PETSc.Mat().create(comm=COMM)
        mat.setSizes([0, 0])
        mat.setFromOptions()
        del mat
    if 'ksp' in ARGS:
        ksp = PETSc.KSP().create(comm=COMM)
        ksp.setFromOptions()
        del ksp
    if 'pc' in ARGS:
        pc = PETSc.PC().create(comm=COMM)
        pc.setFromOptions()
        del pc
    if 'snes' in ARGS:
        snes = PETSc.SNES().create(comm=COMM)
        snes.setFromOptions()
        del snes
    if 'ts' in ARGS:
        ts = PETSc.TS().create(comm=COMM)
        ts.setFromOptions()
        del ts

if __name__ == '__main__':
    help()
