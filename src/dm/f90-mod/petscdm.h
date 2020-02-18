

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
      PetscEnum, parameter :: DM_BOUNDARY_NONE = 0
      PetscEnum, parameter :: DM_BOUNDARY_GHOSTED = 1
      PetscEnum, parameter :: DM_BOUNDARY_MIRROR = 2
      PetscEnum, parameter :: DM_BOUNDARY_PERIODIC = 3
      PetscEnum, parameter :: DM_BOUNDARY_TWIST = 4

!
!  Types of point location
!
      PetscEnum, parameter :: DM_POINTLOCATION_NONE = 0
      PetscEnum, parameter :: DM_POINTLOCATION_NEAREST = 1
      PetscEnum, parameter :: DM_POINTLOCATION_REMOVE = 2

      PetscEnum, parameter :: DM_ADAPT_DETERMINE=-1
      PetscEnum, parameter :: DM_ADAPT_KEEP=0
      PetscEnum, parameter :: DM_ADAPT_REFINE=1
      PetscEnum, parameter :: DM_ADAPT_COARSEN=2
      PetscEnum, parameter :: DM_ADAPT_RESERVED_COUNT=3
!
! DMDA Directions
!
      PetscEnum, parameter :: DM_X = 0
      PetscEnum, parameter :: DM_Y = 1
      PetscEnum, parameter :: DM_Z = 2
