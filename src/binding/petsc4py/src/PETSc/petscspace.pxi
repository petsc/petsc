
cdef extern from * nogil:

    ctypedef const char* PetscSpaceType
    PetscSpaceType PETSCSPACEPOLYNOMIAL
    PetscSpaceType PETSCSPACEPTRIMMED
    PetscSpaceType PETSCSPACETENSOR
    PetscSpaceType PETSCSPACESUM
    PetscSpaceType PETSCSPACEPOINT
    PetscSpaceType PETSCSPACESUBSPACE
    PetscSpaceType PETSCSPACEWXY

    int PetscSpaceCreate(MPI_Comm, PetscSpace*)
    int PetscSpaceSetUp(PetscSpace)
    int PetscSpaceSetFromOptions(PetscSpace)
    int PetscSpaceDestroy(PetscSpace*)
    int PetscSpaceView(PetscSpace, PetscViewer)
    int PetscSpaceSetType(PetscSpace, PetscSpaceType)
    int PetscSpaceGetType(PetscSpace, PetscSpaceType*)
    #int PetscSpaceEvaluate(PetscSpace, PetscInt, const PetscReal [], PetscReal [], PetscReal [])
    int PetscSpaceGetDimension(PetscSpace, PetscInt*)
    int PetscSpaceGetDegree(PetscSpace, PetscInt*, PetscInt*)
    int PetscSpaceGetNumVariables(PetscSpace, PetscInt*)
    int PetscSpaceGetNumComponents(PetscSpace, PetscInt*)
    int PetscSpaceSetDegree(PetscSpace, PetscInt, PetscInt)
    int PetscSpaceSetNumComponents(PetscSpace, PetscInt)
    int PetscSpaceSetNumVariables(PetscSpace, PetscInt)

    int PetscSpaceSumGetConcatenate(PetscSpace, PetscBool*)
    int PetscSpaceSumSetConcatenate(PetscSpace, PetscBool)
    int PetscSpaceSumGetNumSubspaces(PetscSpace, PetscInt*)
    int PetscSpaceSumGetSubspace(PetscSpace, PetscInt, PetscSpace*)
    int PetscSpaceSumSetNumSubspaces(PetscSpace, PetscInt)
    int PetscSpaceSumSetSubspace(PetscSpace,PetscInt, PetscSpace)
    int PetscSpaceTensorGetNumSubspaces(PetscSpace, PetscInt*)
    int PetscSpaceTensorGetSubspace(PetscSpace, PetscInt, PetscSpace*)
    int PetscSpaceTensorSetNumSubspaces(PetscSpace, PetscInt)
    int PetscSpaceTensorSetSubspace(PetscSpace, PetscInt, PetscSpace)
    int PetscSpaceViewFromOptions(PetscSpace, PetscObject, char [])

    int PetscSpacePolynomialSetTensor(PetscSpace, PetscBool)
    int PetscSpacePolynomialGetTensor(PetscSpace, PetscBool*)
    int PetscSpacePointSetPoints(PetscSpace, PetscQuadrature)
    int PetscSpacePointGetPoints(PetscSpace, PetscQuadrature*)

    int PetscSpacePTrimmedSetFormDegree(PetscSpace, PetscInt)
    int PetscSpacePTrimmedGetFormDegree(PetscSpace, PetscInt*)

# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef const char* PetscDualSpaceType
    PetscDualSpaceType PETSCDUALSPACELAGRANGE
    PetscDualSpaceType PETSCDUALSPACESIMPLE
    PetscDualSpaceType PETSCDUALSPACEREFINED
    PetscDualSpaceType PETSCDUALSPACEBDM

    int PetscDualSpaceCreate(MPI_Comm, PetscDualSpace*)
    int PetscDualSpaceDestroy(PetscDualSpace*)
    int PetscDualSpaceDuplicate(PetscDualSpace, PetscDualSpace*)
    int PetscDualSpaceView(PetscDualSpace, PetscViewer)
    
    int PetscDualSpaceGetDM(PetscDualSpace, PetscDM*)
    int PetscDualSpaceSetDM(PetscDualSpace, PetscDM)
    int PetscDualSpaceGetDimension(PetscDualSpace, PetscInt*)
    int PetscDualSpaceGetNumComponents(PetscDualSpace, PetscInt*)
    int PetscDualSpaceSetNumComponents(PetscDualSpace, PetscInt)
    int PetscDualSpaceGetOrder(PetscDualSpace, PetscInt*)
    int PetscDualSpaceSetOrder(PetscDualSpace, PetscInt)
    int PetscDualSpaceGetNumDof(PetscDualSpace, const PetscInt**)
    int PetscDualSpaceSetUp(PetscDualSpace)   
    int PetscDualSpaceViewFromOptions(PetscDualSpace,PetscObject, char[])

    int PetscDualSpaceGetFunctional(PetscDualSpace, PetscInt, PetscQuadrature*)
    int PetscDualSpaceGetInteriorDimension(PetscDualSpace, PetscInt*)
    int PetscDualSpaceLagrangeGetContinuity(PetscDualSpace, PetscBool*)
    int PetscDualSpaceLagrangeGetTensor(PetscDualSpace, PetscBool*)
    int PetscDualSpaceLagrangeGetTrimmed(PetscDualSpace, PetscBool*)
    int PetscDualSpaceLagrangeSetContinuity(PetscDualSpace, PetscBool)
    int PetscDualSpaceLagrangeSetTensor(PetscDualSpace, PetscBool)
    int PetscDualSpaceLagrangeSetTrimmed(PetscDualSpace, PetscBool)
    int PetscDualSpaceSimpleSetDimension(PetscDualSpace, PetscInt)
    int PetscDualSpaceSimpleSetFunctional(PetscDualSpace, PetscInt, PetscQuadrature)
    int PetscDualSpaceGetType(PetscDualSpace, PetscDualSpaceType*)
    int PetscDualSpaceSetType(PetscDualSpace, PetscDualSpaceType)
    
    #int PetscDualSpaceSetFromOptions(PetscDualSpace)
    
    
    #int PetscDualSpaceRefinedSetCellSpaces(PetscDualSpace, const PetscDualSpace [])
    
    # Advanced
    #int PetscDualSpaceCreateAllDataDefault(PetscDualSpace, PetscQuadrature*, PetscMat*)
    #int PetscDualSpaceCreateInteriorDataDefault(PetscDualSpace, PetscQuadrature*, PetscMat*)
    #int PetscDualSpaceEqual(PetscDualSpace, PetscDualSpace, PetscBool*)
    #int PetscDualSpaceGetAllData(PetscDualSpace, PetscQuadrature*, PetscMat*)