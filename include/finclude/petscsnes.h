!
!  Include file for Fortran use of the SNES package in PETSc
!
#include "finclude/petscsnesdef.h"

!
!  Convergence flags
!
      PetscEnum SNES_CONVERGED_FNORM_ABS
      PetscEnum SNES_CONVERGED_FNORM_RELATIVE
      PetscEnum SNES_CONVERGED_SNORM_RELATIVE
      PetscEnum SNES_CONVERGED_ITS
      PetscEnum SNES_CONVERGED_TR_DELTA

      PetscEnum SNES_DIVERGED_FUNCTION_DOMAIN
      PetscEnum SNES_DIVERGED_FUNCTION_COUNT
      PetscEnum SNES_DIVERGED_LINEAR_SOLVE
      PetscEnum SNES_DIVERGED_FNORM_NAN
      PetscEnum SNES_DIVERGED_MAX_IT
      PetscEnum SNES_DIVERGED_LINE_SEARCH
      PetscEnum SNES_DIVERGED_INNER
      PetscEnum SNES_DIVERGED_LOCAL_MIN
      PetscEnum SNES_CONVERGED_ITERATING
   
      parameter (SNES_CONVERGED_FNORM_ABS         =  2)
      parameter (SNES_CONVERGED_FNORM_RELATIVE    =  3)
      parameter (SNES_CONVERGED_SNORM_RELATIVE    =  4)
      parameter (SNES_CONVERGED_ITS               =  5)
      parameter (SNES_CONVERGED_TR_DELTA          =  7)

      parameter (SNES_DIVERGED_FUNCTION_DOMAIN    = -1)
      parameter (SNES_DIVERGED_FUNCTION_COUNT     = -2)  
      parameter (SNES_DIVERGED_LINEAR_SOLVE       = -3)  
      parameter (SNES_DIVERGED_FNORM_NAN          = -4) 
      parameter (SNES_DIVERGED_MAX_IT             = -5)
      parameter (SNES_DIVERGED_LINE_SEARCH        = -6)
      parameter (SNES_DIVERGED_INNER              = -7)
      parameter (SNES_DIVERGED_LOCAL_MIN          = -8)
      parameter (SNES_CONVERGED_ITERATING         =  0)
!
! SNESNormType
!
      PetscEnum SNES_NORM_DEFAULT
      PetscEnum SNES_NORM_NONE
      PetscEnum SNES_NORM_FUNCTION
      PetscEnum SNES_NORM_INITIAL_ONLY
      PetscEnum SNES_NORM_FINAL_ONLY
      PetscEnum SNES_NORM_INITIAL_FINAL_ONLY

      parameter (SNES_NORM_DEFAULT                = -1)
      parameter (SNES_NORM_NONE                   =  0)
      parameter (SNES_NORM_FUNCTION               =  1)
      parameter (SNES_NORM_INITIAL_ONLY           =  2)
      parameter (SNES_NORM_FINAL_ONLY             =  3)
      parameter (SNES_NORM_INITIAL_FINAL_ONLY     =  4)

!
!  Some PETSc fortran functions that the user might pass as arguments
!
      external SNESDEFAULTCOMPUTEJACOBIAN
      external MATMFFDCOMPUTEJACOBIAN
      external SNESDEFAULTCOMPUTEJACOBIANCOLOR
      external SNESMONITORDEFAULT
      external SNESMONITORLG
      external SNESMONITORSOLUTION
      external SNESMONITORSOLUTIONUPDATE

      external SNESDEFAULTCONVERGED
      external SNESSKIPCONVERGED

!
! SNESNGMRESRestartType
!

      PetscEnum SNES_NGMRES_RESTART_NONE
      PetscEnum SNES_NGMRES_RESTART_PERIODIC
      PetscEnum SNES_NGMRES_RESTART_DIFFERENCE

      parameter (SNES_NGMRES_RESTART_NONE = 0)
      parameter (SNES_NGMRES_RESTART_PERIODIC = 1)
      parameter (SNES_NGMRES_RESTART_DIFFERENCE = 2)


!
! SNESNGMRESSelectionType
!

      PetscEnum SNES_NGMRES_SELECT_NONE
      PetscEnum SNES_NGMRES_SELECT_DIFFERENCE
      PetscEnum SNES_NGMRES_SELECT_LINESEARCH

      parameter (SNES_NGMRES_SELECT_NONE = 0)
      parameter (SNES_NGMRES_SELECT_DIFFERENCE = 1)
      parameter (SNES_NGMRES_SELECT_LINESEARCH = 2)


!
! SNESQNCompositionType
!

      PetscEnum SNES_QN_SEQUENTIAL
      PetscEnum SNES_QN_COMPOSED

      parameter (SNES_QN_SEQUENTIAL = 0)
      parameter (SNES_QN_COMPOSED   = 1)

!
! SNESQNScaleType
!

      PetscEnum SNES_QN_SCALE_NONE
      PetscEnum SNES_QN_SCALE_SHANNO
      PetscEnum SNES_QN_SCALE_LINESEARCH
      PetscEnum SNES_QN_SCALE_JACOBIAN

      parameter(SNES_QN_SCALE_NONE       = 0)
      parameter(SNES_QN_SCALE_SHANNO     = 1)
      parameter(SNES_QN_SCALE_LINESEARCH = 2)
      parameter(SNES_QN_SCALE_JACOBIAN   = 3)

!
! SNESQNRestartType
!

      PetscEnum SNES_QN_RESTART_NONE
      PetscEnum SNES_QN_RESTART_POWELL
      PetscEnum SNES_QN_RESTART_PERIODIC

      parameter(SNES_QN_RESTART_NONE     = 0)
      parameter(SNES_QN_RESTART_POWELL   = 1)
      parameter(SNES_QN_RESTART_PERIODIC = 2)

!
! SNESNCGType
!

      PetscEnum SNES_NCG_FR
      PetscEnum SNES_NCG_PRP
      PetscEnum SNES_NCG_HS
      PetscEnum SNES_NCG_DY
      PetscEnum SNES_NCG_CD

      parameter(SNES_NCG_FR  = 0)
      parameter(SNES_NCG_PRP = 1)
      parameter(SNES_NCG_HS  = 2)
      parameter(SNES_NCG_DY  = 3)
      parameter(SNES_NCG_CD  = 4)


!  End of Fortran include file for the SNES package in PETSc

