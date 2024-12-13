!
!  Used by petscdmswarmmod.F90 to create Fortran module file
!
#include "petsc/finclude/petscdmswarm.h"

!
type, extends(tPetscObject) :: tDMSwarmCellDM
      end type tDMSwarmCellDM
      DMSwarmCellDM, parameter :: PETSC_NULL_DM_SWARM_CELL_DM = tDMSwarmCellDM(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_DM_SWARM_CELL_DM
#endif

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
