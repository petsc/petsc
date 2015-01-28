ctypedef int (*PetscDMShellXToYFunction)(PetscDM,
                                         PetscVec,
                                         PetscInsertMode,
                                         PetscVec) except PETSC_ERR_PYTHON
cdef extern from * nogil:
    ctypedef int (*PetscDMShellCreateVectorFunction)(PetscDM,
                                                     PetscVec*) except PETSC_ERR_PYTHON
    ctypedef int (*PetscDMShellCreateMatrixFunction)(PetscDM,
                                                     PetscMat*) except PETSC_ERR_PYTHON
    int DMShellCreate(MPI_Comm,PetscDM*)
    int DMShellSetMatrix(PetscDM,PetscMat)
    int DMShellSetGlobalVector(PetscDM,PetscVec)
    int DMShellSetLocalVector(PetscDM,PetscVec)
    int DMShellSetCreateGlobalVector(PetscDM,PetscDMShellCreateVectorFunction)
    int DMShellSetCreateLocalVector(PetscDM,PetscDMShellCreateVectorFunction)
    int DMShellSetGlobalToLocal(PetscDM,PetscDMShellXToYFunction,PetscDMShellXToYFunction)
    int DMShellSetGlobalToLocalVecScatter(PetscDM,PetscScatter)
    int DMShellSetLocalToGlobal(PetscDM,PetscDMShellXToYFunction,PetscDMShellXToYFunction)
    int DMShellSetLocalToGlobalVecScatter(PetscDM,PetscScatter)
    int DMShellSetLocalToLocal(PetscDM,PetscDMShellXToYFunction,PetscDMShellXToYFunction)
    int DMShellSetLocalToLocalVecScatter(PetscDM,PetscScatter)
    int DMShellSetCreateMatrix(PetscDM,PetscDMShellCreateMatrixFunction)

cdef int DMSHELL_CreateGlobalVector(
    PetscDM dm,
    PetscVec *v) except PETSC_ERR_PYTHON with gil:
    cdef DM Dm = subtype_DM(dm)()
    cdef Vec vec
    Dm.dm = dm
    PetscINCREF(Dm.obj)
    context = Dm.get_attr('__create_global_vector__')
    assert context is not None and type(context) is tuple
    (create_gvec, args, kargs) = context
    vec = create_gvec(Dm, *args, **kargs)
    PetscINCREF(vec.obj)
    v[0] = vec.vec
    return 0

cdef int DMSHELL_CreateLocalVector(
    PetscDM dm,
    PetscVec *v) except PETSC_ERR_PYTHON with gil:
    cdef DM Dm = subtype_DM(dm)()
    cdef Vec vec
    Dm.dm = dm
    PetscINCREF(Dm.obj)
    context = Dm.get_attr('__create_local_vector__')
    assert context is not None and type(context) is tuple
    (create_lvec, args, kargs) = context
    vec = create_lvec(Dm, *args, **kargs)
    PetscINCREF(vec.obj)
    v[0] = vec.vec
    return 0

cdef int DMSHELL_GlobalToLocalBegin(
    PetscDM dm,
    PetscVec g,
    PetscInsertMode mode,
    PetscVec l) except PETSC_ERR_PYTHON with gil:
    cdef DM Dm = subtype_DM(dm)()
    cdef Vec gvec = ref_Vec(g)
    cdef Vec lvec = ref_Vec(l)
    Dm.dm = dm
    PetscINCREF(Dm.obj)
    context = Dm.get_attr('__g2l_begin__')
    assert context is not None and type(context) is tuple
    (begin, args, kargs) = context
    begin(Dm, gvec, mode, lvec, *args, **kargs)
    return 0

cdef int DMSHELL_GlobalToLocalEnd(
    PetscDM dm,
    PetscVec g,
    PetscInsertMode mode,
    PetscVec l) except PETSC_ERR_PYTHON with gil:
    cdef DM Dm = subtype_DM(dm)()
    cdef Vec gvec = ref_Vec(g)
    cdef Vec lvec = ref_Vec(l)
    Dm.dm = dm
    PetscINCREF(Dm.obj)
    context = Dm.get_attr('__g2l_end__')
    assert context is not None and type(context) is tuple
    (end, args, kargs) = context
    end(Dm, gvec, mode, lvec, *args, **kargs)
    return 0

cdef int DMSHELL_LocalToGlobalBegin(
    PetscDM dm,
    PetscVec g,
    PetscInsertMode mode,
    PetscVec l) except PETSC_ERR_PYTHON with gil:
    cdef DM Dm = subtype_DM(dm)()
    cdef Vec gvec = ref_Vec(g)
    cdef Vec lvec = ref_Vec(l)
    Dm.dm = dm
    PetscINCREF(Dm.obj)
    context = Dm.get_attr('__l2g_begin__')
    assert context is not None and type(context) is tuple
    (begin, args, kargs) = context
    begin(Dm, gvec, mode, lvec, *args, **kargs)
    return 0

cdef int DMSHELL_LocalToGlobalEnd(
    PetscDM dm,
    PetscVec g,
    PetscInsertMode mode,
    PetscVec l) except PETSC_ERR_PYTHON with gil:
    cdef DM Dm = subtype_DM(dm)()
    cdef Vec gvec = ref_Vec(g)
    cdef Vec lvec = ref_Vec(l)
    Dm.dm = dm
    PetscINCREF(Dm.obj)
    context = Dm.get_attr('__l2g_end__')
    assert context is not None and type(context) is tuple
    (end, args, kargs) = context
    end(Dm, gvec, mode, lvec, *args, **kargs)
    return 0

cdef int DMSHELL_LocalToLocalBegin(
    PetscDM dm,
    PetscVec g,
    PetscInsertMode mode,
    PetscVec l) except PETSC_ERR_PYTHON with gil:
    cdef DM Dm = subtype_DM(dm)()
    cdef Vec gvec = ref_Vec(g)
    cdef Vec lvec = ref_Vec(l)
    Dm.dm = dm
    PetscINCREF(Dm.obj)
    context = Dm.get_attr('__l2l_begin__')
    assert context is not None and type(context) is tuple
    (begin, args, kargs) = context
    begin(Dm, gvec, mode, lvec, *args, **kargs)
    return 0

cdef int DMSHELL_LocalToLocalEnd(
    PetscDM dm,
    PetscVec g,
    PetscInsertMode mode,
    PetscVec l) except PETSC_ERR_PYTHON with gil:
    cdef DM Dm = subtype_DM(dm)()
    cdef Vec gvec = ref_Vec(g)
    cdef Vec lvec = ref_Vec(l)
    Dm.dm = dm
    PetscINCREF(Dm.obj)
    context = Dm.get_attr('__l2l_end__')
    assert context is not None and type(context) is tuple
    (end, args, kargs) = context
    end(Dm, gvec, mode, lvec, *args, **kargs)
    return 0

cdef int DMSHELL_CreateMatrix(
    PetscDM dm,
    PetscMat *cmat) except PETSC_ERR_PYTHON with gil:
    cdef DM Dm = subtype_DM(dm)()
    cdef Mat mat

    Dm.dm = dm
    PetscINCREF(Dm.obj)
    context = Dm.get_attr('__create_matrix__')
    assert context is not None and type(context) is tuple
    (matrix, args, kargs) = context
    mat = matrix(Dm, *args, **kargs)
    PetscINCREF(mat.obj)
    cmat[0] = mat.mat
    return 0
