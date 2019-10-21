

!
!  Include file for Fortran use of the DM package in PETSc
!
#include "petsc/finclude/petscdm.h"

      type tDM
        sequence
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tDM

      DM, parameter :: PETSC_NULL_DM = tDM(0)
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

!
!  Types of point location
!
      PetscEnum DM_POINTLOCATION_NONE
      PetscEnum DM_POINTLOCATION_NEAREST
      PetscEnum DM_POINTLOCATION_REMOVE

      parameter (DM_POINTLOCATION_NONE = 0)
      parameter (DM_POINTLOCATION_NEAREST = 1)
      parameter (DM_POINTLOCATION_REMOVE = 2)

      PetscEnum DM_ADAPT_DETERMINE
      PetscEnum DM_ADAPT_KEEP
      PetscEnum DM_ADAPT_REFINE
      PetscEnum DM_ADAPT_COARSEN
      PetscEnum DM_ADAPT_RESERVED_COUNT

      parameter (DM_ADAPT_DETERMINE=-1)
      parameter (DM_ADAPT_KEEP=0)
      parameter (DM_ADAPT_REFINE=1)
      parameter (DM_ADAPT_COARSEN=2)
      parameter (DM_ADAPT_RESERVED_COUNT=3)
!
! DMDA Directions
!
      PetscEnum DM_X
      PetscEnum DM_Y
      PetscEnum DM_Z

      parameter (DM_X = 0,DM_Y = 1,DM_Z = 2)
