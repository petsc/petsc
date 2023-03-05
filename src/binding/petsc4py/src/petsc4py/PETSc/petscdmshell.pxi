ctypedef PetscErrorCode (*PetscDMShellXToYFunction)(PetscDM,
                                         PetscVec,
                                         PetscInsertMode,
                                         PetscVec) except PETSC_ERR_PYTHON
cdef extern from * nogil:
    ctypedef PetscErrorCode (*PetscDMShellCreateVectorFunction)(PetscDM,
                                                     PetscVec*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode (*PetscDMShellCreateMatrixFunction)(PetscDM,
                                                     PetscMat*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode (*PetscDMShellTransferFunction)(PetscDM,
                                                 MPI_Comm,
                                                 PetscDM*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode (*PetscDMShellCreateInterpolationFunction)(PetscDM,
                                                            PetscDM,
                                                            PetscMat*,
                                                            PetscVec*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode (*PetscDMShellCreateInjectionFunction)(PetscDM,
                                                        PetscDM,
                                                        PetscMat*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode (*PetscDMShellCreateRestrictionFunction)(PetscDM,
                                                          PetscDM,
                                                          PetscMat*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode (*PetscDMShellCreateFieldDecompositionFunction)(PetscDM,
                                                                 PetscInt*,
                                                                 char***,
                                                                 PetscIS**,
                                                                 PetscDM**) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode (*PetscDMShellCreateDomainDecompositionFunction)(PetscDM,
                                                                 PetscInt*,
                                                                 char***,
                                                                 PetscIS**,
                                                                 PetscIS**,
                                                                 PetscDM**) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode (*PetscDMShellCreateDomainDecompositionScattersFunction)(PetscDM,
                                                                 PetscInt,
                                                                 PetscDM*,
                                                                 PetscScatter**,
                                                                 PetscScatter**,
                                                                 PetscScatter**) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode (*PetscDMShellCreateSubDM)(PetscDM,
                                            PetscInt,
                                            PetscInt[],
                                            PetscIS*,
                                            PetscDM*) except PETSC_ERR_PYTHON
    PetscErrorCode DMShellCreate(MPI_Comm,PetscDM*)
    PetscErrorCode DMShellSetMatrix(PetscDM,PetscMat)
    PetscErrorCode DMShellSetGlobalVector(PetscDM,PetscVec)
    PetscErrorCode DMShellSetLocalVector(PetscDM,PetscVec)
    PetscErrorCode DMShellSetCreateGlobalVector(PetscDM,PetscDMShellCreateVectorFunction)
    PetscErrorCode DMShellSetCreateLocalVector(PetscDM,PetscDMShellCreateVectorFunction)
    PetscErrorCode DMShellSetGlobalToLocal(PetscDM,PetscDMShellXToYFunction,PetscDMShellXToYFunction)
    PetscErrorCode DMShellSetGlobalToLocalVecScatter(PetscDM,PetscScatter)
    PetscErrorCode DMShellSetLocalToGlobal(PetscDM,PetscDMShellXToYFunction,PetscDMShellXToYFunction)
    PetscErrorCode DMShellSetLocalToGlobalVecScatter(PetscDM,PetscScatter)
    PetscErrorCode DMShellSetLocalToLocal(PetscDM,PetscDMShellXToYFunction,PetscDMShellXToYFunction)
    PetscErrorCode DMShellSetLocalToLocalVecScatter(PetscDM,PetscScatter)
    PetscErrorCode DMShellSetCreateMatrix(PetscDM,PetscDMShellCreateMatrixFunction)
    PetscErrorCode DMShellSetCoarsen(PetscDM,PetscDMShellTransferFunction)
    PetscErrorCode DMShellSetRefine(PetscDM,PetscDMShellTransferFunction)
    PetscErrorCode DMShellSetCreateInterpolation(PetscDM,PetscDMShellCreateInterpolationFunction)
    PetscErrorCode DMShellSetCreateInjection(PetscDM,PetscDMShellCreateInjectionFunction)
    PetscErrorCode DMShellSetCreateRestriction(PetscDM,PetscDMShellCreateRestrictionFunction)
    PetscErrorCode DMShellSetCreateFieldDecomposition(PetscDM,PetscDMShellCreateFieldDecompositionFunction)
    PetscErrorCode DMShellSetCreateDomainDecomposition(PetscDM,PetscDMShellCreateDomainDecompositionFunction)
    PetscErrorCode DMShellSetCreateDomainDecompositionScatters(PetscDM,PetscDMShellCreateDomainDecompositionScattersFunction)
    PetscErrorCode DMShellSetCreateSubDM(PetscDM,PetscDMShellCreateSubDM)

    PetscErrorCode VecGetDM(PetscVec,PetscDM*)
    PetscErrorCode VecSetDM(PetscVec,PetscDM)
    PetscErrorCode MatGetDM(PetscMat,PetscDM*)
    PetscErrorCode MatSetDM(PetscMat,PetscDM)


cdef PetscErrorCode DMSHELL_CreateGlobalVector(
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
    cdef PetscDM odm = NULL
    CHKERR( VecGetDM(v[0], &odm) )
    if odm == NULL:
        CHKERR( VecSetDM(v[0], dm) )
    return PETSC_SUCCESS

cdef PetscErrorCode DMSHELL_CreateLocalVector(
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
    cdef PetscDM odm = NULL
    CHKERR( VecGetDM(v[0], &odm) )
    if odm == NULL:
        CHKERR( VecSetDM(v[0], dm) )
    return PETSC_SUCCESS

cdef PetscErrorCode DMSHELL_GlobalToLocalBegin(
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
    return PETSC_SUCCESS

cdef PetscErrorCode DMSHELL_GlobalToLocalEnd(
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
    return PETSC_SUCCESS

cdef PetscErrorCode DMSHELL_LocalToGlobalBegin(
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
    return PETSC_SUCCESS

cdef PetscErrorCode DMSHELL_LocalToGlobalEnd(
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
    return PETSC_SUCCESS

cdef PetscErrorCode DMSHELL_LocalToLocalBegin(
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
    return PETSC_SUCCESS

cdef PetscErrorCode DMSHELL_LocalToLocalEnd(
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
    return PETSC_SUCCESS

cdef PetscErrorCode DMSHELL_CreateMatrix(
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
    cdef PetscDM odm = NULL
    CHKERR( MatGetDM(cmat[0], &odm) )
    if odm == NULL:
        CHKERR( MatSetDM(cmat[0], dm) )
    return PETSC_SUCCESS

cdef PetscErrorCode DMSHELL_Coarsen(
    PetscDM dm,
    MPI_Comm comm,
    PetscDM *dmc) except PETSC_ERR_PYTHON with gil:
    cdef DM Dm = subtype_DM(dm)()
    cdef DM Dmc
    cdef Comm Comm = new_Comm(comm)
    Dm.dm = dm
    PetscINCREF(Dm.obj)
    context = Dm.get_attr('__coarsen__')
    assert context is not None and type(context) is tuple
    (coarsen, args, kargs) = context
    Dmc = coarsen(Dm, Comm, *args, **kargs)
    PetscINCREF(Dmc.obj)
    dmc[0] = Dmc.dm
    return PETSC_SUCCESS

cdef PetscErrorCode DMSHELL_Refine(
    PetscDM dm,
    MPI_Comm comm,
    PetscDM *dmf) except PETSC_ERR_PYTHON with gil:
    cdef DM Dm = subtype_DM(dm)()
    cdef DM Dmf
    cdef Comm Comm = new_Comm(comm)
    Dm.dm = dm
    PetscINCREF(Dm.obj)
    context = Dm.get_attr('__refine__')
    assert context is not None and type(context) is tuple
    (refine, args, kargs) = context
    Dmf = refine(Dm, Comm, *args, **kargs)
    PetscINCREF(Dmf.obj)
    dmf[0] = Dmf.dm
    return PETSC_SUCCESS

cdef PetscErrorCode DMSHELL_CreateInterpolation(
    PetscDM dmc,
    PetscDM dmf,
    PetscMat *cmat,
    PetscVec *cvec) except PETSC_ERR_PYTHON with gil:
    cdef DM Dmc = subtype_DM(dmc)()
    cdef DM Dmf = subtype_DM(dmf)()
    cdef Mat mat
    cdef Vec vec
    Dmc.dm = dmc
    PetscINCREF(Dmc.obj)
    Dmf.dm = dmf
    PetscINCREF(Dmf.obj)
    context = Dmc.get_attr('__create_interpolation__')
    assert context is not None and type(context) is tuple
    (interpolation, args, kargs) = context
    mat, vec = interpolation(Dmc, Dmf, *args, **kargs)
    PetscINCREF(mat.obj)
    cmat[0] = mat.mat
    if cvec == NULL:
        return PETSC_SUCCESS
    if vec is None:
        cvec[0] = NULL
    else:
        PetscINCREF(vec.obj)
        cvec[0] = vec.vec
    return PETSC_SUCCESS

cdef PetscErrorCode DMSHELL_CreateInjection(
    PetscDM dmc,
    PetscDM dmf,
    PetscMat *cmat) except PETSC_ERR_PYTHON with gil:
    cdef DM Dmc = subtype_DM(dmc)()
    cdef DM Dmf = subtype_DM(dmf)()
    cdef Mat mat
    Dmc.dm = dmc
    PetscINCREF(Dmc.obj)
    Dmf.dm = dmf
    PetscINCREF(Dmf.obj)
    context = Dmc.get_attr('__create_injection__')
    assert context is not None and type(context) is tuple
    (injection, args, kargs) = context
    mat = injection(Dmc, Dmf, *args, **kargs)
    PetscINCREF(mat.obj)
    cmat[0] = mat.mat
    return PETSC_SUCCESS

cdef PetscErrorCode DMSHELL_CreateRestriction(
    PetscDM dmf,
    PetscDM dmc,
    PetscMat *cmat) except PETSC_ERR_PYTHON with gil:
    cdef DM Dmf = subtype_DM(dmf)()
    cdef DM Dmc = subtype_DM(dmc)()
    cdef Mat mat
    Dmf.dm = dmf
    PetscINCREF(Dmf.obj)
    Dmc.dm = dmc
    PetscINCREF(Dmc.obj)
    context = Dmf.get_attr('__create_restriction__')
    assert context is not None and type(context) is tuple
    (restriction, args, kargs) = context
    mat = restriction(Dmf, Dmc, *args, **kargs)
    PetscINCREF(mat.obj)
    cmat[0] = mat.mat
    return PETSC_SUCCESS

cdef PetscErrorCode DMSHELL_CreateFieldDecomposition(
    PetscDM dm,
    PetscInt *clen,
    char ***namelist,
    PetscIS **islist,
    PetscDM **dmlist) except PETSC_ERR_PYTHON with gil:
    cdef DM Dm = subtype_DM(dm)()
    cdef int i
    cdef const char *cname = NULL
    Dm.dm = dm
    PetscINCREF(Dm.obj)
    context = Dm.get_attr('__create_field_decomp__')
    assert context is not None and type(context) is tuple
    (decomp, args, kargs) = context
    names, ises, dms = decomp(Dm, *args, **kargs)

    if clen != NULL:
        if names is not None:
            clen[0] = <PetscInt>len(names)
        elif ises is not None:
            clen[0] = <PetscInt>len(ises)
        elif dms is not None:
            clen[0] = <PetscInt>len(dms)
        else:
            clen[0] = 0

    if namelist != NULL and names is not None:
        CHKERR( PetscMalloc(len(names)*sizeof(char**), namelist) )
        for i in range(len(names)):
            names[i] = str2bytes(names[i], &cname)
            CHKERR( PetscStrallocpy(cname, &(namelist[0][i])) )

    if islist != NULL and ises is not None:
        CHKERR( PetscMalloc(len(ises)*sizeof(PetscIS), islist) )
        for i in range(len(ises)):
            islist[0][i] = (<IS?>ises[i]).iset
            PetscINCREF((<IS?>ises[i]).obj)

    if dmlist != NULL and dms is not None:
        CHKERR( PetscMalloc(len(dms)*sizeof(PetscDM), dmlist) )
        for i in range(len(dms)):
            dmlist[0][i] = (<DM?>dms[i]).dm
            PetscINCREF((<DM?>dms[i]).obj)
    return PETSC_SUCCESS

cdef PetscErrorCode DMSHELL_CreateDomainDecomposition(
    PetscDM dm,
    PetscInt *clen,
    char ***namelist,
    PetscIS **innerislist,
    PetscIS **outerislist,
    PetscDM **dmlist) except PETSC_ERR_PYTHON with gil:
    cdef DM Dm = subtype_DM(dm)()
    cdef int i
    cdef const char *cname = NULL
    Dm.dm = dm
    PetscINCREF(Dm.obj)
    context = Dm.get_attr('__create_domain_decomp__')
    assert context is not None and type(context) is tuple
    (decomp, args, kargs) = context
    names, innerises, outerises, dms = decomp(Dm, *args, **kargs)

    if clen != NULL:
        if names is not None:
            clen[0] = <PetscInt>len(names)
        elif innerises is not None:
            clen[0] = <PetscInt>len(innerises)
        elif outerises is not None:
            clen[0] = <PetscInt>len(outerises)
        elif dms is not None:
            clen[0] = <PetscInt>len(dms)
        else:
            clen[0] = 0

    if namelist != NULL and names is not None:
        CHKERR( PetscMalloc(len(names)*sizeof(char**), namelist) )
        for i in range(len(names)):
            names[i] = str2bytes(names[i], &cname)
            CHKERR( PetscStrallocpy(cname, &(namelist[0][i])) )

    if innerislist != NULL and innerises is not None:
        CHKERR( PetscMalloc(len(innerises)*sizeof(PetscIS), innerislist) )
        for i in range(len(innerises)):
            innerislist[0][i] = (<IS?>innerises[i]).iset
            PetscINCREF((<IS?>innerises[i]).obj)

    if outerislist != NULL and outerises is not None:
        CHKERR( PetscMalloc(len(outerises)*sizeof(PetscIS), outerislist) )
        for i in range(len(outerises)):
            outerislist[0][i] = (<IS?>outerises[i]).iset
            PetscINCREF((<IS?>outerises[i]).obj)

    if dmlist != NULL and dms is not None:
        CHKERR( PetscMalloc(len(dms)*sizeof(PetscDM), dmlist) )
        for i in range(len(dms)):
            dmlist[0][i] = (<DM?>dms[i]).dm
            PetscINCREF((<DM?>dms[i]).obj)
    return PETSC_SUCCESS

cdef PetscErrorCode DMSHELL_CreateDomainDecompositionScatters(
    PetscDM dm,
    PetscInt clen,
    PetscDM *subdms,
    PetscScatter** iscat,
    PetscScatter** oscat,
    PetscScatter** gscat) except PETSC_ERR_PYTHON with gil:

    cdef DM Dm = subtype_DM(dm)()
    cdef int i
    cdef const char *cname = NULL
    cdef DM subdm = None

    Dm.dm = dm
    PetscINCREF(Dm.obj)

    psubdms = []
    for i from 0 <= i < clen:
        subdm = subtype_DM(subdms[i])()
        subdm.dm = subdms[i]
        PetscINCREF(subdm.obj)
        psubdms.append(subdm)

    context = Dm.get_attr('__create_domain_decomp_scatters__')
    assert context is not None and type(context) is tuple
    (scatters, args, kargs) = context
    (iscatter, oscatter, gscatter) = scatters(Dm, psubdms, *args, **kargs)

    assert len(iscatter) == clen
    assert len(oscatter) == clen
    assert len(gscatter) == clen

    CHKERR ( PetscMalloc(clen*sizeof(PetscScatter), iscat) )
    CHKERR ( PetscMalloc(clen*sizeof(PetscScatter), oscat) )
    CHKERR ( PetscMalloc(clen*sizeof(PetscScatter), gscat) )

    for i in range(clen):
        iscat[0][i] = (<Scatter?>iscatter[i]).sct
        PetscINCREF((<Scatter?>iscatter[i]).obj)

        oscat[0][i] = (<Scatter?>oscatter[i]).sct
        PetscINCREF((<Scatter?>oscatter[i]).obj)

        gscat[0][i] = (<Scatter?>gscatter[i]).sct
        PetscINCREF((<Scatter?>gscatter[i]).obj)

    return PETSC_SUCCESS

cdef PetscErrorCode DMSHELL_CreateSubDM(
    PetscDM cdm,
    PetscInt numFields,
    const PetscInt cfields[],
    PetscIS *ciset,
    PetscDM *csubdm) except PETSC_ERR_PYTHON with gil:
    cdef DM dm = subtype_DM(cdm)()
    cdef IS iset
    cdef DM subdm
    dm.dm = cdm
    PetscINCREF(dm.obj)
    context = dm.get_attr('__create_subdm__')
    assert context is not None and type(context) is tuple
    (create_subdm, args, kargs) = context

    fields = array_i(numFields, cfields)

    iset, subdm = create_subdm(dm, fields, *args, **kargs)

    PetscINCREF(iset.obj)
    PetscINCREF(subdm.obj)
    ciset[0] = iset.iset
    csubdm[0] = subdm.dm
    return PETSC_SUCCESS
