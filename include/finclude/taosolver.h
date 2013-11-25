#define TaoSolver PetscFortranAddr
#define TaoLineSearch PetscFortranAddr
#define TaoSolverTerminationReason integer

#if !defined (PETSC_AVOID_DECLARATIONS)

      integer TAO_CONVERGED_FATOL
      integer TAO_CONVERGED_FRTOL
      integer TAO_CONVERGED_GATOL
      integer TAO_CONVERGED_GRTOL
      integer TAO_CONVERGED_GTTOL
      integer TAO_CONVERGED_STEPTOL
      integer TAO_CONVERGED_MINF
      integer TAO_CONVERGED_USER
      integer TAO_DIVERGED_MAXITS
      integer TAO_DIVERGED_NAN
      integer TAO_DIVERGED_MAXFCN
      integer TAO_DIVERGED_LS_FAILURE
      integer TAO_DIVERGED_TR_REDUCTION
      integer TAO_DIVERGED_USER
      integer TAO_CONTINUE_ITERATING

      parameter ( TAO_CONVERGED_FATOL = 1)
      parameter ( TAO_CONVERGED_FRTOL = 2)
      parameter ( TAO_CONVERGED_GATOL = 3)
      parameter ( TAO_CONVERGED_GRTOL = 4)
      parameter ( TAO_CONVERGED_GTTOL = 5)
      parameter ( TAO_CONVERGED_STEPTOL = 6)
      parameter ( TAO_CONVERGED_MINF = 7)
      parameter ( TAO_CONVERGED_USER = 8)
      parameter ( TAO_DIVERGED_MAXITS = -2)
      parameter ( TAO_DIVERGED_NAN = -4)
      parameter ( TAO_DIVERGED_MAXFCN = -5)
      parameter ( TAO_DIVERGED_LS_FAILURE = -6)
      parameter ( TAO_DIVERGED_TR_REDUCTION = -7)
      parameter ( TAO_DIVERGED_USER = -8)
      parameter ( TAO_CONTINUE_ITERATING = 0)


#endif
