cdef extern from * nogil:

    ctypedef const char* PetscRegressorType "PetscRegressorType"
    PetscRegressorType PETSCREGRESSORLINEAR

    PetscErrorCode PetscRegressorCreate(MPI_Comm, PetscRegressor*)
    PetscErrorCode PetscRegressorReset(PetscRegressor)
    PetscErrorCode PetscRegressorDestroy(PetscRegressor*)
    PetscErrorCode PetscRegressorSetType(PetscRegressor, PetscRegressorType)
    PetscErrorCode PetscRegressorGetType(PetscRegressor, PetscRegressorType*)
    PetscErrorCode PetscRegressorSetRegularizerWeight(PetscRegressor, PetscReal)
    PetscErrorCode PetscRegressorSetUp(PetscRegressor)
    PetscErrorCode PetscRegressorSetFromOptions(PetscRegressor)
    PetscErrorCode PetscRegressorView(PetscRegressor, PetscViewer)
    PetscErrorCode PetscRegressorFit(PetscRegressor, PetscMat, PetscVec)
    PetscErrorCode PetscRegressorPredict(PetscRegressor, PetscMat, PetscVec)
    PetscErrorCode PetscRegressorGetTao(PetscRegressor, PetscTAO*)

    PetscErrorCode PetscRegressorLinearSetFitIntercept(PetscRegressor, PetscBool)
    PetscErrorCode PetscRegressorLinearSetUseKSP(PetscRegressor, PetscBool)
    PetscErrorCode PetscRegressorLinearGetKSP(PetscRegressor, PetscKSP*)
    PetscErrorCode PetscRegressorLinearGetCoefficients(PetscRegressor, PetscVec*)
    PetscErrorCode PetscRegressorLinearGetIntercept(PetscRegressor, PetscScalar*)
    PetscErrorCode PetscRegressorLinearSetType(PetscRegressor, PetscRegressorLinearType)
    PetscErrorCode PetscRegressorLinearGetType(PetscRegressor, PetscRegressorLinearType*)

    ctypedef enum PetscRegressorLinearType:
        REGRESSOR_LINEAR_OLS
        REGRESSOR_LINEAR_LASSO
        REGRESSOR_LINEAR_RIDGE
