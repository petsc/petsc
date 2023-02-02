
cdef extern from * nogil:

    ctypedef const char* PetscSpaceType
    PetscSpaceType PETSCSPACEPOLYNOMIAL
    PetscSpaceType PETSCSPACEPTRIMMED
    PetscSpaceType PETSCSPACETENSOR
    PetscSpaceType PETSCSPACESUM
    PetscSpaceType PETSCSPACEPOINT
    PetscSpaceType PETSCSPACESUBSPACE
    PetscSpaceType PETSCSPACEWXY

    PetscErrorCode PetscSpaceCreate(MPI_Comm, PetscSpace*)
    PetscErrorCode PetscSpaceSetUp(PetscSpace)
    PetscErrorCode PetscSpaceSetFromOptions(PetscSpace)
    PetscErrorCode PetscSpaceDestroy(PetscSpace*)
    PetscErrorCode PetscSpaceView(PetscSpace, PetscViewer)
    PetscErrorCode PetscSpaceSetType(PetscSpace, PetscSpaceType)
    PetscErrorCode PetscSpaceGetType(PetscSpace, PetscSpaceType*)
    #int PetscSpaceEvaluate(PetscSpace, PetscInt, const PetscReal [], PetscReal [], PetscReal [])
    PetscErrorCode PetscSpaceGetDimension(PetscSpace, PetscInt*)
    PetscErrorCode PetscSpaceGetDegree(PetscSpace, PetscInt*, PetscInt*)
    PetscErrorCode PetscSpaceGetNumVariables(PetscSpace, PetscInt*)
    PetscErrorCode PetscSpaceGetNumComponents(PetscSpace, PetscInt*)
    PetscErrorCode PetscSpaceSetDegree(PetscSpace, PetscInt, PetscInt)
    PetscErrorCode PetscSpaceSetNumComponents(PetscSpace, PetscInt)
    PetscErrorCode PetscSpaceSetNumVariables(PetscSpace, PetscInt)

    PetscErrorCode PetscSpaceSumGetConcatenate(PetscSpace, PetscBool*)
    PetscErrorCode PetscSpaceSumSetConcatenate(PetscSpace, PetscBool)
    PetscErrorCode PetscSpaceSumGetNumSubspaces(PetscSpace, PetscInt*)
    PetscErrorCode PetscSpaceSumGetSubspace(PetscSpace, PetscInt, PetscSpace*)
    PetscErrorCode PetscSpaceSumSetNumSubspaces(PetscSpace, PetscInt)
    PetscErrorCode PetscSpaceSumSetSubspace(PetscSpace,PetscInt, PetscSpace)
    PetscErrorCode PetscSpaceTensorGetNumSubspaces(PetscSpace, PetscInt*)
    PetscErrorCode PetscSpaceTensorGetSubspace(PetscSpace, PetscInt, PetscSpace*)
    PetscErrorCode PetscSpaceTensorSetNumSubspaces(PetscSpace, PetscInt)
    PetscErrorCode PetscSpaceTensorSetSubspace(PetscSpace, PetscInt, PetscSpace)
    PetscErrorCode PetscSpaceViewFromOptions(PetscSpace, PetscObject, char [])

    PetscErrorCode PetscSpacePolynomialSetTensor(PetscSpace, PetscBool)
    PetscErrorCode PetscSpacePolynomialGetTensor(PetscSpace, PetscBool*)
    PetscErrorCode PetscSpacePointSetPoints(PetscSpace, PetscQuadrature)
    PetscErrorCode PetscSpacePointGetPoints(PetscSpace, PetscQuadrature*)

    PetscErrorCode PetscSpacePTrimmedSetFormDegree(PetscSpace, PetscInt)
    PetscErrorCode PetscSpacePTrimmedGetFormDegree(PetscSpace, PetscInt*)

# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef const char* PetscDualSpaceType
    PetscDualSpaceType PETSCDUALSPACELAGRANGE
    PetscDualSpaceType PETSCDUALSPACESIMPLE
    PetscDualSpaceType PETSCDUALSPACEREFINED
    PetscDualSpaceType PETSCDUALSPACEBDM

    PetscErrorCode PetscDualSpaceCreate(MPI_Comm, PetscDualSpace*)
    PetscErrorCode PetscDualSpaceDestroy(PetscDualSpace*)
    PetscErrorCode PetscDualSpaceDuplicate(PetscDualSpace, PetscDualSpace*)
    PetscErrorCode PetscDualSpaceView(PetscDualSpace, PetscViewer)
    
    PetscErrorCode PetscDualSpaceGetDM(PetscDualSpace, PetscDM*)
    PetscErrorCode PetscDualSpaceSetDM(PetscDualSpace, PetscDM)
    PetscErrorCode PetscDualSpaceGetDimension(PetscDualSpace, PetscInt*)
    PetscErrorCode PetscDualSpaceGetNumComponents(PetscDualSpace, PetscInt*)
    PetscErrorCode PetscDualSpaceSetNumComponents(PetscDualSpace, PetscInt)
    PetscErrorCode PetscDualSpaceGetOrder(PetscDualSpace, PetscInt*)
    PetscErrorCode PetscDualSpaceSetOrder(PetscDualSpace, PetscInt)
    PetscErrorCode PetscDualSpaceGetNumDof(PetscDualSpace, const PetscInt**)
    PetscErrorCode PetscDualSpaceSetUp(PetscDualSpace)   
    PetscErrorCode PetscDualSpaceViewFromOptions(PetscDualSpace,PetscObject, char[])

    PetscErrorCode PetscDualSpaceGetFunctional(PetscDualSpace, PetscInt, PetscQuadrature*)
    PetscErrorCode PetscDualSpaceGetInteriorDimension(PetscDualSpace, PetscInt*)
    PetscErrorCode PetscDualSpaceLagrangeGetContinuity(PetscDualSpace, PetscBool*)
    PetscErrorCode PetscDualSpaceLagrangeGetTensor(PetscDualSpace, PetscBool*)
    PetscErrorCode PetscDualSpaceLagrangeGetTrimmed(PetscDualSpace, PetscBool*)
    PetscErrorCode PetscDualSpaceLagrangeSetContinuity(PetscDualSpace, PetscBool)
    PetscErrorCode PetscDualSpaceLagrangeSetTensor(PetscDualSpace, PetscBool)
    PetscErrorCode PetscDualSpaceLagrangeSetTrimmed(PetscDualSpace, PetscBool)
    PetscErrorCode PetscDualSpaceSimpleSetDimension(PetscDualSpace, PetscInt)
    PetscErrorCode PetscDualSpaceSimpleSetFunctional(PetscDualSpace, PetscInt, PetscQuadrature)
    PetscErrorCode PetscDualSpaceGetType(PetscDualSpace, PetscDualSpaceType*)
    PetscErrorCode PetscDualSpaceSetType(PetscDualSpace, PetscDualSpaceType)
    
    #int PetscDualSpaceSetFromOptions(PetscDualSpace)
    
    
    #int PetscDualSpaceRefinedSetCellSpaces(PetscDualSpace, const PetscDualSpace [])
    
    # Advanced
    #int PetscDualSpaceCreateAllDataDefault(PetscDualSpace, PetscQuadrature*, PetscMat*)
    #int PetscDualSpaceCreateInteriorDataDefault(PetscDualSpace, PetscQuadrature*, PetscMat*)
    #int PetscDualSpaceEqual(PetscDualSpace, PetscDualSpace, PetscBool*)
    #int PetscDualSpaceGetAllData(PetscDualSpace, PetscQuadrature*, PetscMat*)