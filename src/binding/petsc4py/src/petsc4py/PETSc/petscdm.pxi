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

    ctypedef PetscErrorCode (*PetscDMCoarsenHook)(PetscDM,
                                       PetscDM,
                                       void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode (*PetscDMRestrictHook)(PetscDM,
                                        PetscMat,
                                        PetscVec,
                                        PetscMat,
                                        PetscDM,
                                        void*) except PETSC_ERR_PYTHON

    PetscErrorCode DMCreate(MPI_Comm,PetscDM*)
    PetscErrorCode DMClone(PetscDM,PetscDM*)
    PetscErrorCode DMDestroy(PetscDM*)
    PetscErrorCode DMView(PetscDM,PetscViewer)
    PetscErrorCode DMLoad(PetscDM,PetscViewer)
    PetscErrorCode DMSetType(PetscDM,PetscDMType)
    PetscErrorCode DMGetType(PetscDM,PetscDMType*)
    PetscErrorCode DMGetDimension(PetscDM,PetscInt*)
    PetscErrorCode DMSetDimension(PetscDM,PetscInt)
    PetscErrorCode DMSetOptionsPrefix(PetscDM,char[])
    PetscErrorCode DMGetOptionsPrefix(PetscDM,char*[])
    PetscErrorCode DMAppendOptionsPrefix(PetscDM,char[])
    PetscErrorCode DMSetFromOptions(PetscDM)
    PetscErrorCode DMViewFromOptions(PetscDM,PetscObject,char[])
    PetscErrorCode DMSetUp(PetscDM)

    PetscErrorCode DMGetAdjacency(PetscDM,PetscInt,PetscBool*,PetscBool*)
    PetscErrorCode DMSetAdjacency(PetscDM,PetscInt,PetscBool,PetscBool)
    PetscErrorCode DMGetBasicAdjacency(PetscDM,PetscBool*,PetscBool*)
    PetscErrorCode DMSetBasicAdjacency(PetscDM,PetscBool,PetscBool)

    PetscErrorCode DMSetNumFields(PetscDM,PetscInt)
    PetscErrorCode DMGetNumFields(PetscDM,PetscInt*)
    PetscErrorCode DMSetField(PetscDM,PetscInt,PetscDMLabel,PetscObject)
    PetscErrorCode DMAddField(PetscDM,PetscDMLabel,PetscObject)
    PetscErrorCode DMGetField(PetscDM,PetscInt,PetscDMLabel*,PetscObject*)
    PetscErrorCode DMClearFields(PetscDM)
    PetscErrorCode DMCopyFields(PetscDM,PetscDM)
    PetscErrorCode DMCreateDS(PetscDM)
    PetscErrorCode DMClearDS(PetscDM)
    PetscErrorCode DMGetDS(PetscDM,PetscDS*)
    PetscErrorCode DMCopyDS(PetscDM,PetscDM)
    PetscErrorCode DMCopyDisc(PetscDM,PetscDM)

    PetscErrorCode DMGetBlockSize(PetscDM,PetscInt*)
    PetscErrorCode DMSetVecType(PetscDM,PetscVecType)
    PetscErrorCode DMCreateLocalVector(PetscDM,PetscVec*)
    PetscErrorCode DMCreateGlobalVector(PetscDM,PetscVec*)
    PetscErrorCode DMGetLocalVector(PetscDM,PetscVec*)
    PetscErrorCode DMRestoreLocalVector(PetscDM,PetscVec*)
    PetscErrorCode DMGetGlobalVector(PetscDM,PetscVec*)
    PetscErrorCode DMRestoreGlobalVector(PetscDM,PetscVec*)
    PetscErrorCode DMSetMatType(PetscDM,PetscMatType)
    PetscErrorCode DMCreateMatrix(PetscDM,PetscMat*)
    PetscErrorCode DMCreateMassMatrix(PetscDM,PetscDM,PetscMat*)

    PetscErrorCode DMGetCoordinateDM(PetscDM,PetscDM*)
    PetscErrorCode DMGetCoordinateSection(PetscDM,PetscSection*)
    PetscErrorCode DMSetCoordinates(PetscDM,PetscVec)
    PetscErrorCode DMGetCoordinates(PetscDM,PetscVec*)
    PetscErrorCode DMSetCoordinatesLocal(PetscDM,PetscVec)
    PetscErrorCode DMGetCoordinatesLocal(PetscDM,PetscVec*)
    PetscErrorCode DMGetCoordinateDim(PetscDM,PetscInt*)
    PetscErrorCode DMSetCoordinateDim(PetscDM,PetscInt)
    PetscErrorCode DMLocalizeCoordinates(PetscDM)
    PetscErrorCode DMProjectCoordinates(PetscDM, PetscFE)

    PetscErrorCode DMCreateInterpolation(PetscDM,PetscDM,PetscMat*,PetscVec*)
    PetscErrorCode DMCreateInjection(PetscDM,PetscDM,PetscMat*)
    PetscErrorCode DMCreateRestriction(PetscDM,PetscDM,PetscMat*)

    PetscErrorCode DMConvert(PetscDM,PetscDMType,PetscDM*)
    PetscErrorCode DMRefine(PetscDM,MPI_Comm,PetscDM*)
    PetscErrorCode DMCoarsen(PetscDM,MPI_Comm,PetscDM*)
    PetscErrorCode DMRefineHierarchy(PetscDM,PetscInt,PetscDM[])
    PetscErrorCode DMCoarsenHierarchy(PetscDM,PetscInt,PetscDM[])
    PetscErrorCode DMGetRefineLevel(PetscDM,PetscInt*)
    PetscErrorCode DMSetRefineLevel(PetscDM,PetscInt)
    PetscErrorCode DMGetCoarsenLevel(PetscDM,PetscInt*)

    PetscErrorCode DMAdaptLabel(PetscDM,PetscDMLabel,PetscDM*)
    PetscErrorCode DMAdaptMetric(PetscDM,PetscVec,PetscDMLabel,PetscDMLabel,PetscDM*)

    PetscErrorCode DMGlobalToLocalBegin(PetscDM,PetscVec,PetscInsertMode,PetscVec)
    PetscErrorCode DMGlobalToLocalEnd(PetscDM,PetscVec,PetscInsertMode,PetscVec)
    PetscErrorCode DMLocalToGlobalBegin(PetscDM,PetscVec,PetscInsertMode,PetscVec)
    PetscErrorCode DMLocalToGlobalEnd(PetscDM,PetscVec,PetscInsertMode,PetscVec)
    PetscErrorCode DMLocalToLocalBegin(PetscDM,PetscVec,PetscInsertMode,PetscVec)
    PetscErrorCode DMLocalToLocalEnd(PetscDM,PetscVec,PetscInsertMode,PetscVec)

    PetscErrorCode DMGetLocalToGlobalMapping(PetscDM,PetscLGMap*)

    PetscErrorCode DMSetSection(PetscDM,PetscSection)
    PetscErrorCode DMGetSection(PetscDM,PetscSection*)
    PetscErrorCode DMSetLocalSection(PetscDM,PetscSection)
    PetscErrorCode DMGetLocalSection(PetscDM,PetscSection*)
    PetscErrorCode DMSetGlobalSection(PetscDM,PetscSection)
    PetscErrorCode DMGetGlobalSection(PetscDM,PetscSection*)
    PetscErrorCode DMCreateSectionSF(PetscDM,PetscSection,PetscSection)
    PetscErrorCode DMGetSectionSF(PetscDM,PetscSF*)
    PetscErrorCode DMSetSectionSF(PetscDM,PetscSF)
    PetscErrorCode DMGetPointSF(PetscDM,PetscSF*)
    PetscErrorCode DMSetPointSF(PetscDM,PetscSF)

    PetscErrorCode DMCreateSubDM(PetscDM, PetscInt, const PetscInt[], PetscIS*, PetscDM*)
    PetscErrorCode DMSetAuxiliaryVec(PetscDM, PetscDMLabel, PetscInt, PetscInt, PetscVec)
    PetscErrorCode DMGetAuxiliaryVec(PetscDM, PetscDMLabel, PetscInt, PetscInt, PetscVec*)

    PetscErrorCode DMCreateLabel(PetscDM,const char[])
    PetscErrorCode DMGetLabelValue(PetscDM,const char[],PetscInt,PetscInt*)
    PetscErrorCode DMSetLabelValue(PetscDM,const char[],PetscInt,PetscInt)
    PetscErrorCode DMHasLabel(PetscDM,const char[],PetscBool*)
    PetscErrorCode DMClearLabelValue(PetscDM,const char[],PetscInt,PetscInt)
    PetscErrorCode DMGetLabelSize(PetscDM,const char[],PetscInt*)
    PetscErrorCode DMGetLabelIdIS(PetscDM,const char[],PetscIS*)
    PetscErrorCode DMGetStratumSize(PetscDM,const char[],PetscInt,PetscInt*)
    PetscErrorCode DMGetStratumIS(PetscDM,const char[],PetscInt,PetscIS*)
    PetscErrorCode DMClearLabelStratum(PetscDM,const char[],PetscInt)
    PetscErrorCode DMSetLabelOutput(PetscDM,const char[],PetscBool)
    PetscErrorCode DMGetLabelOutput(PetscDM,const char[],PetscBool*)
    PetscErrorCode DMGetNumLabels(PetscDM,PetscInt*)
    PetscErrorCode DMGetLabelName(PetscDM,PetscInt,const char**)
    PetscErrorCode DMHasLabel(PetscDM,const char[],PetscBool*)
    PetscErrorCode DMGetLabel(PetscDM,const char*,PetscDMLabel*)
    PetscErrorCode DMAddLabel(PetscDM,PetscDMLabel)
    PetscErrorCode DMRemoveLabel(PetscDM,const char[],PetscDMLabel*)
    PetscErrorCode DMLabelDestroy(PetscDMLabel *)
    #int DMCopyLabels(PetscDM,PetscDM)

    PetscErrorCode DMShellSetGlobalVector(PetscDM,PetscVec)
    PetscErrorCode DMShellSetLocalVector(PetscDM,PetscVec)

    PetscErrorCode DMKSPSetComputeOperators(PetscDM,PetscKSPComputeOpsFunction,void*)

    PetscErrorCode DMCreateFieldDecomposition(PetscDM,PetscInt*,char***,PetscIS**,PetscDM**)

    PetscErrorCode DMSNESSetFunction(PetscDM,PetscSNESFunctionFunction,void*)
    PetscErrorCode DMSNESSetJacobian(PetscDM,PetscSNESJacobianFunction,void*)

    PetscErrorCode DMCoarsenHookAdd(PetscDM,PetscDMCoarsenHook,PetscDMRestrictHook,void*)

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

cdef PetscErrorCode DM_PyCoarsenHook(
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
    return PETSC_SUCCESS

cdef PetscErrorCode DM_PyRestrictHook(
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
    return PETSC_SUCCESS
# -----------------------------------------------------------------------------
