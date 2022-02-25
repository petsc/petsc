!
! Used by petsckspdefmod.F90 to create Fortran module file
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
      PetscEnum, parameter :: PC_LEFT=0
      PetscEnum, parameter :: PC_RIGHT=1
      PetscEnum, parameter :: PC_SYMMETRIC=2
!
!     PCJacobiType
!
      PetscEnum, parameter :: PC_JACOBI_DIAGONAL=0
      PetscEnum, parameter :: PC_JACOBI_ROWMAX=1
      PetscEnum, parameter :: PC_JACOBI_ROWSUM=2
!
! PCASMType
!
      PetscEnum, parameter :: PC_ASM_BASIC = 3
      PetscEnum, parameter :: PC_ASM_RESTRICT = 1
      PetscEnum, parameter :: PC_ASM_INTERPOLATE = 2
      PetscEnum, parameter :: PC_ASM_NONE = 0
!
! PCCompositeType
!
      PetscEnum, parameter :: PC_COMPOSITE_ADDITIVE=0
      PetscEnum, parameter :: PC_COMPOSITE_MULTIPLICATIVE=1
      PetscEnum, parameter :: PC_COMPOSITE_SYM_MULTIPLICATIVE=2
      PetscEnum, parameter :: PC_COMPOSITE_SPECIAL=3
      PetscEnum, parameter :: PC_COMPOSITE_SCHUR=4
!
! PCRichardsonConvergedReason
!
      PetscEnum, parameter :: PCRICHARDSON_CONVERGED_RTOL = 2
      PetscEnum, parameter :: PCRICHARDSON_CONVERGED_ATOL = 3
      PetscEnum, parameter :: PCRICHARDSON_CONVERGED_ITS  = 4
      PetscEnum, parameter :: PCRICHARDSON_DIVERGED_DTOL = -4
!
! PCFieldSplitSchurPreType
!
      PetscEnum, parameter :: PC_FIELDSPLIT_SCHUR_PRE_SELF=0
      PetscEnum, parameter :: PC_FIELDSPLIT_SCHUR_PRE_SELFP=1
      PetscEnum, parameter :: PC_FIELDSPLIT_SCHUR_PRE_A11=2
      PetscEnum, parameter :: PC_FIELDSPLIT_SCHUR_PRE_USER=3
      PetscEnum, parameter :: PC_FIELDSPLIT_SCHUR_PRE_FULL=4
!
! PCPARMSGlobalType
!
      PetscEnum, parameter :: PC_PARMS_GLOBAL_RAS=0
      PetscEnum, parameter :: PC_PARMS_GLOBAL_SCHUR=1
      PetscEnum, parameter :: PC_PARMS_GLOBAL_BJ=2
!
! PCPARMSLocalType
!
      PetscEnum, parameter :: PC_PARMS_LOCAL_ILU0=0
      PetscEnum, parameter :: PC_PARMS_LOCAL_ILUK=1
      PetscEnum, parameter :: PC_PARMS_LOCAL_ILUT=2
      PetscEnum, parameter :: PC_PARMS_LOCAL_ARMS=3
!
! PCFieldSplitSchurFactType
!
      PetscEnum, parameter :: PC_FIELDSPLIT_SCHUR_FACT_DIAG=0
      PetscEnum, parameter :: PC_FIELDSPLIT_SCHUR_FACT_LOWER=1
      PetscEnum, parameter :: PC_FIELDSPLIT_SCHUR_FACT_UPPER=2
      PetscEnum, parameter :: PC_FIELDSPLIT_SCHUR_FACT_FULL=3

!
! CoarseProblemType
!
      PetscEnum, parameter :: SEQUENTIAL_BDDC=0
      PetscEnum, parameter :: REPLICATED_BDDC=1
      PetscEnum, parameter :: PARALLEL_BDDC=2
      PetscEnum, parameter :: MULTILEVEL_BDDC=3

      PetscEnum, parameter :: PC_MG_MULTIPLICATIVE=0
      PetscEnum, parameter :: PC_MG_ADDITIVE=1
      PetscEnum, parameter :: PC_MG_FULL=2
      PetscEnum, parameter :: PC_MG_KASKADE=3
      PetscEnum, parameter :: PC_MG_CASCADE=3

! PCMGCycleType
      PetscEnum, parameter :: PC_MG_CYCLE_V = 1
      PetscEnum, parameter :: PC_MG_CYCLE_W = 2

! PCMGGalerkinType
      PetscEnum, parameter :: PC_MG_GALERKIN_BOTH = 0
      PetscEnum, parameter :: PC_MG_GALERKIN_PMAT = 1
      PetscEnum, parameter :: PC_MG_GALERKIN_MAT = 2
      PetscEnum, parameter :: PC_MG_GALERKIN_NONE = 3
      PetscEnum, parameter :: PC_MG_GALERKIN_EXTERNAL = 4

      PetscEnum, parameter :: PC_EXOTIC_FACE=0
      PetscEnum, parameter :: PC_EXOTIC_WIREBASKET=1

! PCDeflationSpaceType
      PetscEnum, parameter :: PC_DEFLATION_SPACE_HAAR = 0
      PetscEnum, parameter :: PC_DEFLATION_SPACE_DB2  = 1
      PetscEnum, parameter :: PC_DEFLATION_SPACE_DB4  = 2
      PetscEnum, parameter :: PC_DEFLATION_SPACE_DB8  = 3
      PetscEnum, parameter :: PC_DEFLATION_SPACE_DB16 = 4
      PetscEnum, parameter :: PC_DEFLATION_SPACE_BIORTH22 = 5
      PetscEnum, parameter :: PC_DEFLATION_SPACE_MEYER = 6
      PetscEnum, parameter :: PC_DEFLATION_SPACE_AGGREGATION = 7
      PetscEnum, parameter :: PC_DEFLATION_SPACE_USER = 8
! PCBDDCInterfaceExtType
      PetscEnum, parameter :: PC_BDDC_INTERFACE_EXT_DIRICHLET=0
      PetscEnum, parameter :: PC_BDDC_INTERFACE_EXT_LUMP=1
! PCHPDDMCoarseCorrectionType
      PetscEnum, parameter :: PC_HPDDM_COARSE_CORRECTION_DEFLATED=0
      PetscEnum, parameter :: PC_HPDDM_COARSE_CORRECTION_ADDITIVE=1
      PetscEnum, parameter :: PC_HPDDM_COARSE_CORRECTION_BALANCED=2
!
! PCFailedReason
!
      PetscEnum, parameter :: PC_NOERROR=0
      PetscEnum, parameter :: PC_FACTOR_STRUCT_ZEROPIVOT=1
      PetscEnum, parameter :: PC_FACTOR_NUMERIC_ZEROPIVOT=2
      PetscEnum, parameter :: PC_FACTOR_OUTMEMORY=3
      PetscEnum, parameter :: PC_FACTOR_OTHER=4
      PetscEnum, parameter :: PC_SUBPC_ERROR=5

      external  PCMGRESIDUALDEFAULT

