!
!
!  Include file for Fortran use of the PC (preconditioner) package in PETSc
!
#include "petsc/finclude/petscpc.h"

      type tPC
        sequence
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tPC

      PC, parameter :: PETSC_NULL_PC = tPC(0)
!
!  PCSide
!
      PetscEnum PC_LEFT
      PetscEnum PC_RIGHT
      PetscEnum PC_SYMMETRIC
      parameter (PC_LEFT=0,PC_RIGHT=1,PC_SYMMETRIC=2)
!
!     PCJacobiType
!
      PetscEnum PC_JACOBI_DIAGONAL
      PetscEnum PC_JACOBI_ROWMAX
      PetscEnum PC_JACOBI_ROWSUM
      parameter (PC_JACOBI_DIAGONAL=0)
      parameter (PC_JACOBI_ROWMAX=1)
      parameter (PC_JACOBI_ROWSUM=2)
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
      PetscEnum PC_FIELDSPLIT_SCHUR_PRE_SELFP
      PetscEnum PC_FIELDSPLIT_SCHUR_PRE_A11
      PetscEnum PC_FIELDSPLIT_SCHUR_PRE_USER
      PetscEnum PC_FIELDSPLIT_SCHUR_PRE_FULL
      parameter (PC_FIELDSPLIT_SCHUR_PRE_SELF=0)
      parameter (PC_FIELDSPLIT_SCHUR_PRE_SELFP=1)
      parameter (PC_FIELDSPLIT_SCHUR_PRE_A11=2)
      parameter (PC_FIELDSPLIT_SCHUR_PRE_USER=3)
      parameter (PC_FIELDSPLIT_SCHUR_PRE_FULL=4)
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

      PetscEnum PC_MG_MULTIPLICATIVE
      PetscEnum PC_MG_ADDITIVE
      PetscEnum PC_MG_FULL
      PetscEnum PC_MG_KASKADE
      PetscEnum PC_MG_CASCADE
      parameter (PC_MG_MULTIPLICATIVE=0,PC_MG_ADDITIVE=1)
      parameter (PC_MG_FULL=2,PC_MG_KASKADE=3)
      parameter (PC_MG_CASCADE=3)

! PCMGCycleType
      PetscEnum PC_MG_CYCLE_V
      PetscEnum PC_MG_CYCLE_W
      parameter (PC_MG_CYCLE_V = 1,PC_MG_CYCLE_W = 2)

! PCMGGalerkinType
      PetscEnum PC_MG_GALERKIN_BOTH
      PetscEnum PC_MG_GALERKIN_PMAT
      PetscEnum PC_MG_GALERKIN_MAT
      PetscEnum PC_MG_GALERKIN_NONE
      PetscEnum PC_MG_GALERKIN_EXTERNAL
      parameter (PC_MG_GALERKIN_BOTH = 0)
      parameter (PC_MG_GALERKIN_PMAT = 1)
      parameter (PC_MG_GALERKIN_MAT = 2)
      parameter (PC_MG_GALERKIN_NONE = 3)
      parameter (PC_MG_GALERKIN_EXTERNAL = 4)

      PetscEnum PC_EXOTIC_FACE
      PetscEnum PC_EXOTIC_WIREBASKET
      parameter (PC_EXOTIC_FACE=0,PC_EXOTIC_WIREBASKET=1)

! PCDeflationSpaceType
      PetscEnum PC_DEFLATION_SPACE_HAAR
      PetscEnum PC_DEFLATION_SPACE_DB2
      PetscEnum PC_DEFLATION_SPACE_DB4
      PetscEnum PC_DEFLATION_SPACE_DB8
      PetscEnum PC_DEFLATION_SPACE_DB16
      PetscEnum PC_DEFLATION_SPACE_BIORTH22
      PetscEnum PC_DEFLATION_SPACE_MEYER
      PetscEnum PC_DEFLATION_SPACE_AGGREGATION
      PetscEnum PC_DEFLATION_SPACE_USER
      parameter (PC_DEFLATION_SPACE_HAAR = 0)
      parameter (PC_DEFLATION_SPACE_DB2  = 1)
      parameter (PC_DEFLATION_SPACE_DB4  = 2)
      parameter (PC_DEFLATION_SPACE_DB8  = 3)
      parameter (PC_DEFLATION_SPACE_DB16 = 4)
      parameter (PC_DEFLATION_SPACE_BIORTH22 = 5)
      parameter (PC_DEFLATION_SPACE_MEYER = 6)
      parameter (PC_DEFLATION_SPACE_AGGREGATION = 7)
      parameter (PC_DEFLATION_SPACE_USER = 8)
! PCBDDCInterfaceExtType
      PetscEnum PC_BDDC_INTERFACE_EXT_DIRICHLET
      PetscEnum PC_BDDC_INTERFACE_EXT_LUMP
      parameter (PC_BDDC_INTERFACE_EXT_DIRICHLET=0)
      parameter (PC_BDDC_INTERFACE_EXT_LUMP=1)

!
! PCFailedReason
!
      PetscEnum PC_NOERROR
      PetscEnum PC_FACTOR_STRUCT_ZEROPIVOT
      PetscEnum PC_FACTOR_NUMERIC_ZEROPIVOT
      PetscEnum PC_FACTOR_OUTMEMORY
      PetscEnum PC_FACTOR_OTHER
      PetscEnum PC_SUBPC_ERROR

      parameter (PC_NOERROR=0)
      parameter (PC_FACTOR_STRUCT_ZEROPIVOT=1)
      parameter (PC_FACTOR_NUMERIC_ZEROPIVOT=2)
      parameter (PC_FACTOR_OUTMEMORY=3)
      parameter (PC_FACTOR_OTHER=4)
      parameter (PC_SUBPC_ERROR=5)

      external  PCMGRESIDUALDEFAULT
!
!  End of Fortran include file for the PC package in PETSc

