!
!  Used by petsckspdefmod.F90 to create Fortran module file
!
#include "petsc/finclude/petscksp.h"

      type tKSP
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tKSP

      KSP, parameter :: PETSC_NULL_KSP = tKSP(0)

      type tKSPGuess
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tKSPGuess

      KSPGuess, parameter :: PETSC_NULL_KSPGuess = tKSPGuess(0)

      PetscEnum, parameter :: KSP_CG_SYMMETRIC=0
      PetscEnum, parameter :: KSP_CG_HERMITIAN=1

      PetscEnum, parameter :: MAT_SCHUR_COMPLEMENT_AINV_DIAG=0
      PetscEnum, parameter :: MAT_SCHUR_COMPLEMENT_AINV_LUMP=1
      PetscEnum, parameter :: MAT_SCHUR_COMPLEMENT_AINV_BLOCK_DIAG=2

      PetscEnum, parameter :: KSP_FCD_TRUNC_TYPE_STANDARD=0
      PetscEnum, parameter :: KSP_FCD_TRUNC_TYPE_NOTAY=1

      PetscEnum, parameter :: KSP_CONVERGED_RTOL            = 2
      PetscEnum, parameter :: KSP_CONVERGED_ATOL            = 3
      PetscEnum, parameter :: KSP_CONVERGED_ITS             = 4
      PetscEnum, parameter :: KSP_CONVERGED_CG_NEG_CURVE    = 5
      PetscEnum, parameter :: KSP_CONVERGED_CG_CONSTRAINED  = 6
      PetscEnum, parameter :: KSP_CONVERGED_STEP_LENGTH     = 7
      PetscEnum, parameter :: KSP_CONVERGED_HAPPY_BREAKDOWN = 8

      PetscEnum, parameter :: KSP_DIVERGED_NULL           = -2
      PetscEnum, parameter :: KSP_DIVERGED_ITS            = -3
      PetscEnum, parameter :: KSP_DIVERGED_DTOL           = -4
      PetscEnum, parameter :: KSP_DIVERGED_BREAKDOWN      = -5
      PetscEnum, parameter :: KSP_DIVERGED_BREAKDOWN_BICG = -6
      PetscEnum, parameter :: KSP_DIVERGED_NONSYMMETRIC   = -7
      PetscEnum, parameter :: KSP_DIVERGED_INDEFINITE_PC  = -8
      PetscEnum, parameter :: KSP_DIVERGED_NANORINF       = -9
      PetscEnum, parameter :: KSP_DIVERGED_INDEFINITE_MAT = -10
      PetscEnum, parameter :: KSP_DIVERGED_PC_FAILED = -11

      PetscEnum, parameter :: KSP_CONVERGED_ITERATING = 0

!
!  MatLMVMSymBrdnScaleType
!
      PetscEnum, parameter :: MAT_LMVM_SYMBRDN_SCALE_NONE     = 0
      PetscEnum, parameter :: MAT_LMVM_SYMBRDN_SCALE_SCALAR   = 1
      PetscEnum, parameter :: MAT_LMVM_SYMBRDN_SCALE_DIAGONAL = 2
      PetscEnum, parameter :: MAT_LMVM_SYMBRDN_SCALE_USER     = 3

!
!  Possible arguments to KSPSetNormType()
!
      PetscEnum, parameter :: KSP_NORM_DEFAULT=0
      PetscEnum, parameter :: KSP_NORM_NONE=0
      PetscEnum, parameter :: KSP_NORM_PRECONDITIONED=1
      PetscEnum, parameter :: KSP_NORM_UNPRECONDITIONED=2
      PetscEnum, parameter :: KSP_NORM_NATURAL=3
!
!   Possible arguments to KSPMonitorSet()
!
      external KSPCONVERGEDDEFAULT
      external KSPMONITORRESIDUAL
      external KSPMONITORTRUERESIDUAL
      external KSPMONITORSOLUTION
      external KSPMONITORSINGULARVALUE
      external KSPGMRESMONITORKRYLOV
      external KSPGMRESCLASSICALGRAMSCHMIDTORTHOGONALIZATION
      external KSPGMRESMODIFIEDGRAMSCHMIDTORTHOGONALIZATION
!
!   Possible arguments to KSPGMRESSetRefinementType()
!
      PetscEnum, parameter :: KSP_GMRES_CGS_REFINE_NEVER = 0
      PetscEnum, parameter :: KSP_GMRES_CGS_REFINE_IFNEEDED = 1
      PetscEnum, parameter :: KSP_GMRES_CGS_REFINE_ALWAYS = 2
