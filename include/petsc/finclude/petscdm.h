

!
!  Include file for Fortran use of the DM package in PETSc
!
#include "petsc/finclude/petscdmdef.h"

!
!  Types of periodicity
!
      PetscEnum DM_BOUNDARY_NONE
      PetscEnum DM_BOUNDARY_GHOSTED
      PetscEnum DM_BOUNDARY_MIRROR
      PetscEnum DM_BOUNDARY_PERIODIC
      PetscEnum DM_BOUNDARY_TWIST

      parameter (DM_BOUNDARY_NONE = 0)
      parameter (DM_BOUNDARY_GHOSTED = 1)
      parameter (DM_BOUNDARY_MIRROR = 2)
      parameter (DM_BOUNDARY_PERIODIC = 3)
      parameter (DM_BOUNDARY_TWIST = 4)

