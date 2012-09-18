!
!
!  Include file for Fortran use of the PC (preconditioner) package in PETSc
!
#include "finclude/petscpcdef.h"
!
!  PCSide
!
      PetscEnum PC_LEFT
      PetscEnum PC_RIGHT
      PetscEnum PC_SYMMETRIC
      parameter (PC_LEFT=0,PC_RIGHT=1,PC_SYMMETRIC=2)

      PetscEnum USE_PRECONDITIONER_MATRIX
      PetscEnum USE_TRUE_MATRIX
      parameter (USE_PRECONDITIONER_MATRIX=0,USE_TRUE_MATRIX=1)

!
! PCASMType
!
      PetscEnum PC_ASM_BASIC
      PetscEnum PC_ASM_RESTRICT
      PetscEnum PC_ASM_INTERPOLATE
      PetscEnum PC_ASM_NONE

      parameter (PC_ASM_BASIC = 3,PC_ASM_RESTRICT = 1)
      parameter (PC_ASM_INTERPOLATE = 2,PC_ASM_NONE = 0)
!
! PCCompositeType
!
      PetscEnum PC_COMPOSITE_ADDITIVE
      PetscEnum PC_COMPOSITE_MULTIPLICATIVE
      PetscEnum PC_COMPOSITE_SYM_MULTIPLICATIVE
      PetscEnum PC_COMPOSITE_SPECIAL
      PetscEnum PC_COMPOSITE_SCHUR
      parameter (PC_COMPOSITE_ADDITIVE=0,PC_COMPOSITE_MULTIPLICATIVE=1)
      parameter (PC_COMPOSITE_SYM_MULTIPLICATIVE=2)
      parameter (PC_COMPOSITE_SPECIAL=3,PC_COMPOSITE_SCHUR=4)
!
! PCRichardsonConvergedReason
!
      PetscEnum PCRICHARDSON_CONVERGED_RTOL
      PetscEnum PCRICHARDSON_CONVERGED_ATOL
      PetscEnum PCRICHARDSON_CONVERGED_ITS
      PetscEnum PCRICHARDSON_DIVERGED_DTOL
      parameter (PCRICHARDSON_CONVERGED_RTOL = 2)
      parameter (PCRICHARDSON_CONVERGED_ATOL = 3)
      parameter (PCRICHARDSON_CONVERGED_ITS  = 4)
      parameter (PCRICHARDSON_DIVERGED_DTOL = -4)
!
! PCFieldSplitSchurPreType
!
      PetscEnum PC_FIELDSPLIT_SCHUR_PRE_SELF
      PetscEnum PC_FIELDSPLIT_SCHUR_PRE_DIAG
      PetscEnum PC_FIELDSPLIT_SCHUR_PRE_USER
      parameter (PC_FIELDSPLIT_SCHUR_PRE_SELF=0)
      parameter (PC_FIELDSPLIT_SCHUR_PRE_DIAG=1)
      parameter (PC_FIELDSPLIT_SCHUR_PRE_USER=2)
!
! PCPARMSGlobalType
!
      PetscEnum PC_PARMS_GLOBAL_RAS
      PetscEnum PC_PARMS_GLOBAL_SCHUR
      PetscEnum PC_PARMS_GLOBAL_BJ
      parameter (PC_PARMS_GLOBAL_RAS=0)
      parameter (PC_PARMS_GLOBAL_SCHUR=1)
      parameter (PC_PARMS_GLOBAL_BJ=2)
!
! PCPARMSLocalType
!
      PetscEnum PC_PARMS_LOCAL_ILU0
      PetscEnum PC_PARMS_LOCAL_ILUK
      PetscEnum PC_PARMS_LOCAL_ILUT
      PetscEnum PC_PARMS_LOCAL_ARMS
      parameter (PC_PARMS_LOCAL_ILU0=0)
      parameter (PC_PARMS_LOCAL_ILUK=1)
      parameter (PC_PARMS_LOCAL_ILUT=2)
      parameter (PC_PARMS_LOCAL_ARMS=3)
!
! PCFieldSplitSchurFactType
!
      PetscEnum PC_FIELDSPLIT_SCHUR_FACT_DIAG
      PetscEnum PC_FIELDSPLIT_SCHUR_FACT_LOWER
      PetscEnum PC_FIELDSPLIT_SCHUR_FACT_UPPER
      PetscEnum PC_FIELDSPLIT_SCHUR_FACT_FULL
      parameter (PC_FIELDSPLIT_SCHUR_FACT_DIAG=0)
      parameter (PC_FIELDSPLIT_SCHUR_FACT_LOWER=1)
      parameter (PC_FIELDSPLIT_SCHUR_FACT_UPPER=2)
      parameter (PC_FIELDSPLIT_SCHUR_FACT_FULL=3)

!
! CoarseProblemType
!
      PetscEnum SEQUENTIAL_BDDC
      PetscEnum REPLICATED_BDDC
      PetscEnum PARALLEL_BDDC
      PetscEnum MULTILEVEL_BDDC
      parameter (SEQUENTIAL_BDDC=0)
      parameter (REPLICATED_BDDC=1)
      parameter (PARALLEL_BDDC=2)
      parameter (MULTILEVEL_BDDC=3)
!
!  End of Fortran include file for the PC package in PETSc

