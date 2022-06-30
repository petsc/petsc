# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef const char* PetscDMType "DMType"
    PetscDMType DMDA_type "DMDA"
    PetscDMType DMCOMPOSITE
    PetscDMType DMSLICED
    PetscDMType DMSHELL
    PetscDMType DMPLEX
    PetscDMType DMREDUNDANT
    PetscDMType DMPATCH
    PetscDMType DMMOAB
    PetscDMType DMNETWORK
    PetscDMType DMFOREST
    PetscDMType DMP4EST
    PetscDMType DMP8EST
    PetscDMType DMSWARM
    PetscDMType DMPRODUCT
    PetscDMType DMSTAG

    ctypedef enum PetscDMBoundaryType"DMBoundaryType":
        DM_BOUNDARY_NONE
        DM_BOUNDARY_GHOSTED
        DM_BOUNDARY_MIRROR
        DM_BOUNDARY_PERIODIC
        DM_BOUNDARY_TWIST

    ctypedef enum PetscDMPolytopeType "DMPolytopeType":
        DM_POLYTOPE_POINT
        DM_POLYTOPE_SEGMENT
        DM_POLYTOPE_POINT_PRISM_TENSOR
        DM_POLYTOPE_TRIANGLE
        DM_POLYTOPE_QUADRILATERAL
        DM_POLYTOPE_SEG_PRISM_TENSOR
        DM_POLYTOPE_TETRAHEDRON
        DM_POLYTOPE_HEXAHEDRON
        DM_POLYTOPE_TRI_PRISM
        DM_POLYTOPE_TRI_PRISM_TENSOR
        DM_POLYTOPE_QUAD_PRISM_TENSOR
        DM_POLYTOPE_PYRAMID
        DM_POLYTOPE_FV_GHOST
        DM_POLYTOPE_INTERIOR_GHOST
        DM_POLYTOPE_UNKNOWN
        DM_NUM_POLYTOPES

    ctypedef int (*PetscDMCoarsenHook)(PetscDM,
                                       PetscDM,
                                       void*) except PETSC_ERR_PYTHON
    ctypedef int (*PetscDMRestrictHook)(PetscDM,
                                        PetscMat,
                                        PetscVec,
                                        PetscMat,
                                        PetscDM,
                                        void*) except PETSC_ERR_PYTHON

    int DMCreate(MPI_Comm,PetscDM*)
    int DMClone(PetscDM,PetscDM*)
    int DMDestroy(PetscDM*)
    int DMView(PetscDM,PetscViewer)
    int DMLoad(PetscDM,PetscViewer)
    int DMSetType(PetscDM,PetscDMType)
    int DMGetType(PetscDM,PetscDMType*)
    int DMGetDimension(PetscDM,PetscInt*)
    int DMSetDimension(PetscDM,PetscInt)
    int DMSetOptionsPrefix(PetscDM,char[])
    int DMGetOptionsPrefix(PetscDM,char*[])
    int DMAppendOptionsPrefix(PetscDM,char[])
    int DMSetFromOptions(PetscDM)
    int DMViewFromOptions(PetscDM,PetscObject,char[])
    int DMSetUp(PetscDM)

    int DMGetAdjacency(PetscDM,PetscInt,PetscBool*,PetscBool*)
    int DMSetAdjacency(PetscDM,PetscInt,PetscBool,PetscBool)
    int DMGetBasicAdjacency(PetscDM,PetscBool*,PetscBool*)
    int DMSetBasicAdjacency(PetscDM,PetscBool,PetscBool)

    int DMSetNumFields(PetscDM,PetscInt)
    int DMGetNumFields(PetscDM,PetscInt*)
    int DMSetField(PetscDM,PetscInt,PetscDMLabel,PetscObject)
    int DMAddField(PetscDM,PetscDMLabel,PetscObject)
    int DMGetField(PetscDM,PetscInt,PetscDMLabel*,PetscObject*)
    int DMClearFields(PetscDM)
    int DMCopyFields(PetscDM,PetscDM)
    int DMCreateDS(PetscDM)
    int DMClearDS(PetscDM)
    int DMGetDS(PetscDM,PetscDS*)
    int DMCopyDS(PetscDM,PetscDM)
    int DMCopyDisc(PetscDM,PetscDM)

    int DMGetBlockSize(PetscDM,PetscInt*)
    int DMSetVecType(PetscDM,PetscVecType)
    int DMCreateLocalVector(PetscDM,PetscVec*)
    int DMCreateGlobalVector(PetscDM,PetscVec*)
    int DMGetLocalVector(PetscDM,PetscVec*)
    int DMRestoreLocalVector(PetscDM,PetscVec*)
    int DMGetGlobalVector(PetscDM,PetscVec*)
    int DMRestoreGlobalVector(PetscDM,PetscVec*)
    int DMSetMatType(PetscDM,PetscMatType)
    int DMCreateMatrix(PetscDM,PetscMat*)
    int DMCreateMassMatrix(PetscDM,PetscDM,PetscMat*)

    int DMGetCoordinateDM(PetscDM,PetscDM*)
    int DMGetCoordinateSection(PetscDM,PetscSection*)
    int DMSetCoordinates(PetscDM,PetscVec)
    int DMGetCoordinates(PetscDM,PetscVec*)
    int DMSetCoordinatesLocal(PetscDM,PetscVec)
    int DMGetCoordinatesLocal(PetscDM,PetscVec*)
    int DMGetCoordinateDim(PetscDM,PetscInt*)
    int DMSetCoordinateDim(PetscDM,PetscInt)
    int DMLocalizeCoordinates(PetscDM)
    int DMProjectCoordinates(PetscDM, PetscFE)

    int DMCreateInterpolation(PetscDM,PetscDM,PetscMat*,PetscVec*)
    int DMCreateInjection(PetscDM,PetscDM,PetscMat*)
    int DMCreateRestriction(PetscDM,PetscDM,PetscMat*)

    int DMConvert(PetscDM,PetscDMType,PetscDM*)
    int DMRefine(PetscDM,MPI_Comm,PetscDM*)
    int DMCoarsen(PetscDM,MPI_Comm,PetscDM*)
    int DMRefineHierarchy(PetscDM,PetscInt,PetscDM[])
    int DMCoarsenHierarchy(PetscDM,PetscInt,PetscDM[])
    int DMGetRefineLevel(PetscDM,PetscInt*)
    int DMSetRefineLevel(PetscDM,PetscInt)
    int DMGetCoarsenLevel(PetscDM,PetscInt*)

    int DMAdaptLabel(PetscDM,PetscDMLabel,PetscDM*)
    int DMAdaptMetric(PetscDM,PetscVec,PetscDMLabel,PetscDMLabel,PetscDM*)

    int DMGlobalToLocalBegin(PetscDM,PetscVec,PetscInsertMode,PetscVec)
    int DMGlobalToLocalEnd(PetscDM,PetscVec,PetscInsertMode,PetscVec)
    int DMLocalToGlobalBegin(PetscDM,PetscVec,PetscInsertMode,PetscVec)
    int DMLocalToGlobalEnd(PetscDM,PetscVec,PetscInsertMode,PetscVec)
    int DMLocalToLocalBegin(PetscDM,PetscVec,PetscInsertMode,PetscVec)
    int DMLocalToLocalEnd(PetscDM,PetscVec,PetscInsertMode,PetscVec)

    int DMGetLocalToGlobalMapping(PetscDM,PetscLGMap*)

    int DMSetSection(PetscDM,PetscSection)
    int DMGetSection(PetscDM,PetscSection*)
    int DMSetGlobalSection(PetscDM,PetscSection)
    int DMGetGlobalSection(PetscDM,PetscSection*)
    int DMCreateSectionSF(PetscDM,PetscSection,PetscSection)
    int DMGetSectionSF(PetscDM,PetscSF*)
    int DMSetSectionSF(PetscDM,PetscSF)
    int DMGetPointSF(PetscDM,PetscSF*)
    int DMSetPointSF(PetscDM,PetscSF)

    int DMCreateSubDM(PetscDM, PetscInt, const PetscInt[], PetscIS*, PetscDM*)
    int DMSetAuxiliaryVec(PetscDM, PetscDMLabel, PetscInt, PetscInt, PetscVec)
    int DMGetAuxiliaryVec(PetscDM, PetscDMLabel, PetscInt, PetscInt, PetscVec*)

    int DMCreateLabel(PetscDM,const char[])
    int DMGetLabelValue(PetscDM,const char[],PetscInt,PetscInt*)
    int DMSetLabelValue(PetscDM,const char[],PetscInt,PetscInt)
    int DMHasLabel(PetscDM,const char[],PetscBool*)
    int DMClearLabelValue(PetscDM,const char[],PetscInt,PetscInt)
    int DMGetLabelSize(PetscDM,const char[],PetscInt*)
    int DMGetLabelIdIS(PetscDM,const char[],PetscIS*)
    int DMGetStratumSize(PetscDM,const char[],PetscInt,PetscInt*)
    int DMGetStratumIS(PetscDM,const char[],PetscInt,PetscIS*)
    int DMClearLabelStratum(PetscDM,const char[],PetscInt)
    int DMSetLabelOutput(PetscDM,const char[],PetscBool)
    int DMGetLabelOutput(PetscDM,const char[],PetscBool*)
    int DMGetNumLabels(PetscDM,PetscInt*)
    int DMGetLabelName(PetscDM,PetscInt,const char**)
    int DMHasLabel(PetscDM,const char[],PetscBool*)
    int DMGetLabel(PetscDM,const char*,PetscDMLabel*)
    int DMAddLabel(PetscDM,PetscDMLabel)
    int DMRemoveLabel(PetscDM,const char[],PetscDMLabel*)
    int DMLabelDestroy(PetscDMLabel *)
    #int DMCopyLabels(PetscDM,PetscDM)

    int DMShellSetGlobalVector(PetscDM,PetscVec)
    int DMShellSetLocalVector(PetscDM,PetscVec)

    int DMKSPSetComputeOperators(PetscDM,PetscKSPComputeOpsFunction,void*)

    int DMCreateFieldDecomposition(PetscDM,PetscInt*,char***,PetscIS**,PetscDM**)

    int DMSNESSetFunction(PetscDM,PetscSNESFunctionFunction,void*)
    int DMSNESSetJacobian(PetscDM,PetscSNESJacobianFunction,void*)

    int DMCoarsenHookAdd(PetscDM,PetscDMCoarsenHook,PetscDMRestrictHook,void*)

# --------------------------------------------------------------------

cdef inline PetscDMBoundaryType asBoundaryType(object boundary) \
    except <PetscDMBoundaryType>(-1):
    if boundary is None:
        return DM_BOUNDARY_NONE
    if boundary is False:
        return DM_BOUNDARY_NONE
    if boundary is True:
        return DM_BOUNDARY_PERIODIC
    if isinstance(boundary, str):
        if boundary == 'none':
            return DM_BOUNDARY_NONE
        elif boundary == 'ghosted':
            return DM_BOUNDARY_GHOSTED
        elif boundary == 'mirror':
            return DM_BOUNDARY_MIRROR
        elif boundary == 'periodic':
            return DM_BOUNDARY_PERIODIC
        elif boundary == 'twist':
            return DM_BOUNDARY_TWIST
        else:
            raise ValueError("unknown boundary type: %s" % boundary)
    return boundary

cdef inline PetscInt asBoundary(object boundary,
                                PetscDMBoundaryType *_x,
                                PetscDMBoundaryType *_y,
                                PetscDMBoundaryType *_z) except -1:
    cdef PetscInt dim = 0
    cdef object x=None, y=None, z=None
    if (boundary is None or
        isinstance(boundary, str) or
        isinstance(boundary, int)):
        _x[0] = _y[0] = _z[0] = asBoundaryType(boundary)
    else:
        _x[0] = _y[0] = _z[0] = DM_BOUNDARY_NONE
        boundary = tuple(boundary)
        dim = <PetscInt>len(boundary)
        if   dim == 0: pass
        elif dim == 1: (x,) = boundary
        elif dim == 2: (x, y) = boundary
        elif dim == 3: (x, y, z) = boundary
        if dim >= 1: _x[0] = asBoundaryType(x)
        if dim >= 2: _y[0] = asBoundaryType(y)
        if dim >= 3: _z[0] = asBoundaryType(z)
    return dim

cdef inline object toBoundary(PetscInt dim,
                              PetscDMBoundaryType x,
                              PetscDMBoundaryType y,
                              PetscDMBoundaryType z):
    if   dim == 0: return ()
    elif dim == 1: return (x,)
    elif dim == 2: return (x, y)
    elif dim == 3: return (x, y, z)

# -----------------------------------------------------------------------------

cdef inline DM ref_DM(PetscDM dm):
    cdef DM ob = <DM> DM()
    ob.dm = dm
    PetscINCREF(ob.obj)
    return ob

# --------------------------------------------------------------------

cdef int DM_PyCoarsenHook(
    PetscDM fine,
    PetscDM coarse,
    void*   ctx,
    ) except PETSC_ERR_PYTHON with gil:

    cdef DM Fine = ref_DM(fine)
    cdef DM Coarse = ref_DM(coarse)
    cdef object hooks = Fine.get_attr('__coarsenhooks__')
    assert hooks is not None and type(hooks) is list
    for hook in hooks:
        (hookop, args, kargs) = hook
        hookop(Fine, Coarse, *args, **kargs)
    return 0

cdef int DM_PyRestrictHook(
    PetscDM fine,
    PetscMat mrestrict,
    PetscVec rscale,
    PetscMat inject,
    PetscDM coarse,
    void*   ctx,
    ) except PETSC_ERR_PYTHON with gil:

    cdef DM  Fine = ref_DM(fine)
    cdef Mat Mrestrict = ref_Mat(mrestrict)
    cdef Vec Rscale = ref_Vec(rscale)
    cdef Mat Inject = ref_Mat(inject)
    cdef DM  Coarse = ref_DM(coarse)
    cdef object hooks = Fine.get_attr('__restricthooks__')
    assert hooks is not None and type(hooks) is list
    for hook in hooks:
        (hookop, args, kargs) = hook
        hookop(Fine, Mrestrict, Rscale, Inject, Coarse, *args, **kargs)
    return 0
# -----------------------------------------------------------------------------
