class RegressorType(object):
    """REGRESSOR solver type.

    See Also
    --------
    petsc.PetscRegressorType

    """
    LINEAR = S_(PETSCREGRESSORLINEAR)


class RegressorLinearType(object):
    """Linear regressor type.

    See Also
    --------
    petsc.PetscRegressorLinearType
    """
    OLS   = REGRESSOR_LINEAR_OLS
    LASSO = REGRESSOR_LINEAR_LASSO
    RIDGE = REGRESSOR_LINEAR_RIDGE


cdef class Regressor(Object):
    """Regression solver.

    REGRESSOR  is described in the `PETSc manual <petsc:manual/regressor>`.

    See Also
    --------
    petsc.PetscRegressor

    """

    Type = RegressorType
    LinearType = RegressorLinearType

    def __cinit__(self):
        self.obj = <PetscObject*> &self.regressor
        self.regressor = NULL

    def view(self, Viewer viewer=None) -> None:
        """View the solver.

        Collective.

        Parameters
        ----------
        viewer
            A `Viewer` instance or `None` for the default viewer.

        See Also
        --------
        petsc.PetscRegressorView

        """
        cdef PetscViewer cviewer = NULL
        if viewer is not None: cviewer = viewer.vwr
        CHKERR(PetscRegressorView(self.regressor, cviewer))

    def create(self, comm=None) -> Self:
        """Create a REGRESSOR solver.

        Collective.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        Sys.getDefaultComm, petsc.PetscRegressorCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscRegressor newregressor = NULL
        CHKERR(PetscRegressorCreate(ccomm, &newregressor))
        PetscCLEAR(self.obj); self.regressor = newregressor
        return self

    def setRegularizerWeight(self, weight: float) -> None:
        """Set the weight to be used for the regularizer.

        Logically collective.

        See Also
        --------
        setType, petsc.PetscRegressorSetRegularizerWeight
        """
        CHKERR(PetscRegressorSetRegularizerWeight(self.regressor, weight))

    def setFromOptions(self) -> None:
        """Configure the solver from the options database.

        Collective.

        See Also
        --------
        petsc_options, petsc.PetscRegressorSetFromOptions
        """
        CHKERR(PetscRegressorSetFromOptions(self.regressor))

    def setUp(self) -> None:
        """Set up the internal data structures for using the solver.

        Collective.

        See Also
        --------
        petsc.PetscRegressorSetUp

        """
        CHKERR(PetscRegressorSetUp(self.regressor))

    def fit(self, Mat X, Vec y) -> None:
        """Fit the regression problem.

        Collective.

        Parameters
        ----------
        X
            The matrix of training data
        y
            The vector of target values from the training dataset

        See Also
        --------
        petsc.PetscRegressorPredict

        """
        CHKERR(PetscRegressorFit(self.regressor, X.mat, y.vec))

    def predict(self, Mat X, Vec y) -> None:
        """Predict the regression problem.

        Collective.

        Parameters
        ----------
        X
            The matrix of unlabeled observations
        y
            The vector of predicted labels

        See Also
        --------
        petsc.PetscRegressorFit

        """
        CHKERR(PetscRegressorPredict(self.regressor, X.mat, y.vec))

    def getTAO(self) -> TAO:
        """Return the underlying `TAO` object .

        Not collective.

        See Also
        --------
        getLinearKSP, petsc.PetscRegressorGetTao
        """
        cdef TAO tao = TAO()
        CHKERR(PetscRegressorGetTao(self.regressor, &tao.tao))
        CHKERR(PetscINCREF(tao.obj))
        return tao

    def reset(self) -> None:
        """Destroy internal data structures of the solver.

        Collective.

        See Also
        --------
        petsc.PetscRegressorDestroy

        """
        CHKERR(PetscRegressorReset(self.regressor))

    def destroy(self) -> Self:
        """Destroy the solver.

        Collective.

        See Also
        --------
        petsc.PetscRegressorDestroy

        """
        CHKERR(PetscRegressorDestroy(&self.regressor))
        return self

    def setType(self, regressor_type: Type | str) -> None:
        """Set the type of the solver.

        Logically collective.

        Parameters
        ----------
        regressor_type
            The type of the solver.

        See Also
        --------
        getType, petsc.PetscRegressorSetType

        """
        cdef PetscRegressorType cval = NULL
        regressor_type = str2bytes(regressor_type, &cval)
        CHKERR(PetscRegressorSetType(self.regressor, cval))

    def getType(self) -> str:
        """Return the type of the solver.

        Not collective.

        See Also
        --------
        setType, petsc.PetscRegressorGetType

        """
        cdef PetscRegressorType ctype = NULL
        CHKERR(PetscRegressorGetType(self.regressor, &ctype))
        return bytes2str(ctype)

    # --- Linear ---

    def setLinearFitIntercept(self, flag: bool) -> None:
        """Set a flag to indicate that the intercept should be calculated.

        Logically collective.

        See Also
        --------
        petsc.PetscRegressorLinearSetFitIntercept
        """
        cdef PetscBool fitintercept = flag
        CHKERR(PetscRegressorLinearSetFitIntercept(self.regressor, fitintercept))

    def setLinearUseKSP(self, flag: bool) -> None:
        """Set a flag to indicate that `KSP` instead of `TAO` solvers should be used.

        Logically collective.

        See Also
        --------
        petsc.PetscRegressorLinearSetUseKSP
        """
        cdef PetscBool useksp = flag
        CHKERR(PetscRegressorLinearSetUseKSP(self.regressor, useksp))

    def getLinearKSP(self) -> KSP:
        """Returns the `KSP` context used by the linear regressor.

        Not collective.

        See Also
        --------
        petsc.PetscRegressorLinearGetKSP
        """
        cdef KSP ksp = KSP()
        CHKERR(PetscRegressorLinearGetKSP(self.regressor, &ksp.ksp))
        CHKERR(PetscINCREF(ksp.obj))
        return ksp

    def getLinearCoefficients(self) -> Vec:
        """Get a vector of the fitted coefficients from a linear regression model.

        Not collective.

        See Also
        --------
        getLinearIntercept, petsc.PetscRegressorLinearGetCoefficients
        """
        cdef Vec coeffs = Vec()
        CHKERR(PetscRegressorLinearGetCoefficients(self.regressor, &coeffs.vec))
        CHKERR(PetscINCREF(coeffs.obj))
        return coeffs

    def getLinearIntercept(self) -> Scalar:
        """Get the intercept from a linear regression model.

        Not collective.

        See Also
        --------
        setLinearFitIntercept, petsc.PetscRegressorLinearGetIntercept
        """
        cdef PetscScalar intercept = 0.0
        CHKERR(PetscRegressorLinearGetIntercept(self.regressor, &intercept))
        return toScalar(intercept)

    def setLinearType(self, lineartype: RegressorLinearType) -> None:
        """Set the type of linear regression to be performed.

        Logically collective.

        See Also
        --------
        getLinearType, petsc.PetscRegressorLinearSetType
        """
        CHKERR(PetscRegressorLinearSetType(self.regressor, lineartype))

    def getLinearType(self) -> RegressorLinearType:
        """Return the type of the linear regressor.

        Not collective.

        See Also
        --------
        setLinearType, petsc.PetscRegressorLinearGetType
        """
        cdef PetscRegressorLinearType cval = REGRESSOR_LINEAR_OLS
        CHKERR(PetscRegressorLinearGetType(self.regressor, &cval))
        return cval

del RegressorType
