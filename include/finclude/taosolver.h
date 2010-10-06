#define TaoSolver PetscFortranAddr
#define TaoSolverTerminationReason integer

#if !defined (PETSC_AVOID_DECLARATIONS)

      integer TAO_CONVERGED_ATOL
      integer TAO_CONVERGED_RTOL
      integer TAO_CONVERGED_TRTOL
      integer TAO_CONVERGED_MINF
      integer TAO_CONVERGED_USER
      integer TAO_DIVERGED_MAXITS
      integer TAO_DIVERGED_NAN
      integer TAO_DIVERGED_MAXFCN
      integer TAO_DIVERGED_LS_FAILURE
      integer TAO_DIVERGED_TR_REDUCTION
      integer TAO_DIVERGED_USER
      integer TAO_CONTINUE_ITERATING

      parameter ( TAO_CONVERGED_ATOL = 2)
      parameter ( TAO_CONVERGED_RTOL = 3)
      parameter ( TAO_CONVERGED_TRTOL = 4)
      parameter ( TAO_CONVERGED_MINF = 5)
      parameter ( TAO_CONVERGED_USER = 6)
      parameter ( TAO_DIVERGED_MAXITS = -2)
      parameter ( TAO_DIVERGED_NAN = -4)
      parameter ( TAO_DIVERGED_MAXFCN = -5)
      parameter ( TAO_DIVERGED_LS_FAILURE = -6)
      parameter ( TAO_DIVERGED_TR_REDUCTION = -7)
      parameter ( TAO_DIVERGED_USER = -8)
      parameter ( TAO_CONTINUE_ITERATING = 0)


#endif
