# --------------------------------------------------------------------

cdef extern from * nogil:
  
    ctypedef const char* PetscFEType
    PetscFEType PETSCFEBASIC
    PetscFEType PETSCFEOPENCL
    PetscFEType PETSCFECOMPOSITE

    PetscErrorCode PetscFECreate(MPI_Comm, PetscFE*)
    PetscErrorCode PetscFECreateDefault(MPI_Comm, PetscInt, PetscInt, PetscBool, const char [], PetscInt, PetscFE*)
    PetscErrorCode PetscFECreateLagrange(MPI_Comm, PetscInt, PetscInt, PetscBool, PetscInt, PetscInt, PetscFE*)
    PetscErrorCode PetscFESetType(PetscFE, PetscFEType)
    PetscErrorCode PetscFEGetQuadrature(PetscFE, PetscQuadrature*)
    PetscErrorCode PetscFEGetFaceQuadrature(PetscFE, PetscQuadrature*)
    PetscErrorCode PetscFESetQuadrature(PetscFE, PetscQuadrature)
    PetscErrorCode PetscFESetFaceQuadrature(PetscFE, PetscQuadrature)
    PetscErrorCode PetscFEDestroy(PetscFE*)
    PetscErrorCode PetscFEGetBasisSpace(PetscFE, PetscSpace*)
    PetscErrorCode PetscFESetBasisSpace(PetscFE, PetscSpace)
    PetscErrorCode PetscFEGetDimension(PetscFE, PetscInt*)
    PetscErrorCode PetscFEGetNumComponents(PetscFE, PetscInt*)
    PetscErrorCode PetscFESetNumComponents(PetscFE, PetscInt)
    PetscErrorCode PetscFEGetNumDof(PetscFE, const PetscInt**)
    PetscErrorCode PetscFEGetSpatialDimension(PetscFE, PetscInt*)
    PetscErrorCode PetscFEGetTileSizes(PetscFE, PetscInt*, PetscInt*, PetscInt*, PetscInt*)
    PetscErrorCode PetscFESetTileSizes(PetscFE, PetscInt, PetscInt, PetscInt, PetscInt)
    PetscErrorCode PetscFEGetDualSpace(PetscFE, PetscDualSpace*)
    PetscErrorCode PetscFESetDualSpace(PetscFE, PetscDualSpace)
    PetscErrorCode PetscFESetFromOptions(PetscFE)
    PetscErrorCode PetscFESetUp(PetscFE)

    PetscErrorCode PetscFEView(PetscFE, PetscViewer)
    PetscErrorCode PetscFEViewFromOptions(PetscFE, PetscObject, char[])
    
