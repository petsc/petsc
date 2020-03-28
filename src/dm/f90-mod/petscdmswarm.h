!
!  Include file for Fortran use of the DMSwarm package in PETSc
!
#include "petsc/finclude/petscdmswarm.h"

!
! DMSwarmType
!
      PetscEnum, parameter :: DMSWARM_BASIC = 0
      PetscEnum, parameter :: DMSWARM_PIC = 1
!
! DMSwarmMigrateType
!
      PetscEnum, parameter :: DMSWARM_MIGRATE_BASIC = 0
      PetscEnum, parameter :: DMSWARM_MIGRATE_DMCELLNSCATTER = 1
      PetscEnum, parameter :: DMSWARM_MIGRATE_DMCELLEXACT = 2
      PetscEnum, parameter :: DMSWARM_MIGRATE_USER = 3
!
! DMSwarmCollectType
!
      PetscEnum, parameter :: DMSWARM_COLLECT_BASIC = 0
      PetscEnum, parameter :: DMSWARM_COLLECT_DMDABOUNDINGBOX = 1
      PetscEnum, parameter :: DMSWARM_COLLECT_GENERAL = 2
      PetscEnum, parameter :: DMSWARM_COLLECT_USER = 3
!
! DMSwarmPICLayoutType
!
      PetscEnum, parameter :: DMSWARMPIC_LAYOUT_REGULAR = 0
      PetscEnum, parameter :: DMSWARMPIC_LAYOUT_GAUSS = 1
      PetscEnum, parameter :: DMSWARMPIC_LAYOUT_SUBDIVISION = 2
