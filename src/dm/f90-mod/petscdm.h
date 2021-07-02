!
! Used by petscdmmod.F90 to create Fortran module file
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
!
! Polytope types
!
      PetscEnum, parameter :: DM_POLYTOPE_POINT = 0
      PetscEnum, parameter :: DM_POLYTOPE_SEGMENT = 1
      PetscEnum, parameter :: DM_POLYTOPE_POINT_PRISM_TENSOR = 2
      PetscEnum, parameter :: DM_POLYTOPE_TRIANGLE = 3
      PetscEnum, parameter :: DM_POLYTOPE_QUADRILATERAL = 4
      PetscEnum, parameter :: DM_POLYTOPE_SEG_PRISM_TENSOR = 5
      PetscEnum, parameter :: DM_POLYTOPE_TETRAHEDRON = 6
      PetscEnum, parameter :: DM_POLYTOPE_HEXAHEDRON = 7
      PetscEnum, parameter :: DM_POLYTOPE_TRI_PRISM = 8
      PetscEnum, parameter :: DM_POLYTOPE_TRI_PRISM_TENSOR = 9
      PetscEnum, parameter :: DM_POLYTOPE_QUAD_PRISM_TENSOR = 10
      PetscEnum, parameter :: DM_POLYTOPE_PYRAMID = 11
      PetscEnum, parameter :: DM_POLYTOPE_FV_GHOST = 12
      PetscEnum, parameter :: DM_POLYTOPE_INTERIOR_GHOST = 13
      PetscEnum, parameter :: DM_POLYTOPE_UNKNOWN = 14
      PetscEnum, parameter :: DM_NUM_POLYTOPES = 15

      type tDMPlexCellRefiner
        sequence
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tDMPlexCellRefiner

      DMPlexCellRefiner, parameter :: PETSC_NULL_DMPLEXCELLREFINER = tDMPlexCellRefiner(0)
