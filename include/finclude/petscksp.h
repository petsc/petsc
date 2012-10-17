!
!
!  Include file for Fortran use of the KSP package in PETSc
!
#include "finclude/petsckspdef.h"

!
!  CG Types
!
      PetscEnum KSP_CG_SYMMETRIC
      PetscEnum KSP_CG_HERMITIAN

      parameter (KSP_CG_SYMMETRIC=0,KSP_CG_HERMITIAN=1)

      PetscEnum KSP_CONVERGED_RTOL
      PetscEnum KSP_CONVERGED_ATOL
      PetscEnum KSP_CONVERGED_ITS
      PetscEnum KSP_DIVERGED_NULL
      PetscEnum KSP_DIVERGED_ITS
      PetscEnum KSP_DIVERGED_DTOL
      PetscEnum KSP_DIVERGED_BREAKDOWN
      PetscEnum KSP_CONVERGED_ITERATING
      PetscEnum KSP_CONVERGED_CG_NEG_CURVE
      PetscEnum KSP_CONVERGED_CG_CONSTRAINED
      PetscEnum KSP_CONVERGED_STEP_LENGTH
      PetscEnum KSP_CONVERGED_HAPPY_BREAKDOWN
      PetscEnum KSP_DIVERGED_BREAKDOWN_BICG
      PetscEnum KSP_DIVERGED_NONSYMMETRIC
      PetscEnum KSP_DIVERGED_INDEFINITE_PC
      PetscEnum KSP_DIVERGED_NAN
      PetscEnum KSP_DIVERGED_INDEFINITE_MAT

      parameter (KSP_CONVERGED_RTOL            = 2)
      parameter (KSP_CONVERGED_ATOL            = 3)
      parameter (KSP_CONVERGED_ITS             = 4)
      parameter (KSP_CONVERGED_CG_NEG_CURVE    = 5)
      parameter (KSP_CONVERGED_CG_CONSTRAINED  = 6)
      parameter (KSP_CONVERGED_STEP_LENGTH     = 7)
      parameter (KSP_CONVERGED_HAPPY_BREAKDOWN = 8)

      parameter (KSP_DIVERGED_NULL           = -2)
      parameter (KSP_DIVERGED_ITS            = -3)
      parameter (KSP_DIVERGED_DTOL           = -4)
      parameter (KSP_DIVERGED_BREAKDOWN      = -5)
      parameter (KSP_DIVERGED_BREAKDOWN_BICG = -6)
      parameter (KSP_DIVERGED_NONSYMMETRIC   = -7)
      parameter (KSP_DIVERGED_INDEFINITE_PC  = -8)
      parameter (KSP_DIVERGED_NAN            = -9)
      parameter (KSP_DIVERGED_INDEFINITE_MAT = -10)

      parameter (KSP_CONVERGED_ITERATING = 0)
!
!  Possible arguments to KSPSetNormType()
!
      PetscEnum KSP_NORM_NONE
      PetscEnum KSP_NORM_PRECONDITIONED
      PetscEnum KSP_NORM_UNPRECONDITIONED
      PetscEnum KSP_NORM_NATURAL

      parameter (KSP_NORM_NONE=0)
      parameter (KSP_NORM_PRECONDITIONED=1)
      parameter (KSP_NORM_UNPRECONDITIONED=2)
      parameter (KSP_NORM_NATURAL=3)
!
!   Possible arguments to KSPMonitorSet()
!
      external KSPDEFAULTCONVERGED
      external KSPMONITORDEFAULT
      external KSPMONITORTRUERESIDUALNORM
      external KSPMONITORLGRESIDUALNORM
      external KSPMONITORLGTRUERESIDUALNORM
      external KSPMONITORSOLUTION
      external KSPMONITORSINGULARVALUE
      external KSPGMRESMONITORKRYLOV
!
!   Possible arguments to KSPGMRESSetRefinementType()
!
      PetscEnum KSP_GMRES_CGS_REFINE_NEVER
      PetscEnum KSP_GMRES_CGS_REFINE_IFNEEDED
      PetscEnum KSP_GMRES_CGS_REFINE_ALWAYS
!
      parameter (KSP_GMRES_CGS_REFINE_NEVER = 0)
      parameter (KSP_GMRES_CGS_REFINE_IFNEEDED = 1)
      parameter (KSP_GMRES_CGS_REFINE_ALWAYS = 2)
!
!  End of Fortran include file for the KSP package in PETSc
!

