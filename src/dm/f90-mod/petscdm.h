!
! Used by petscdmmod.F90 to create Fortran module file
!
#include "petsc/finclude/petscdm.h"

      type, extends(tPetscObject) :: tDM
      end type tDM
      type, extends(tPetscObject) :: tDMAdaptor
      end type tDMAdaptor
      type, extends(tPetscObject) :: tDMField
      end type tDMField
      type, extends(tPetscObject) :: tPetscQuadrature
      end type tPetscQuadrature
      type, extends(tPetscObject) :: tPetscWeakForm
      end type tPetscWeakForm
      type, extends(tPetscObject) :: tPetscDS
      end type tPetscDS
      type, extends(tPetscObject) :: tPetscFE
      end type tPetscFE
      type, extends(tPetscObject) :: tPetscSpace
      end type tPetscSpace
      type, extends(tPetscObject) :: tPetscDualSpace
      end type tPetscDualSpace
      type, extends(tPetscObject) :: tPetscFV
      end type tPetscFV
      type, extends(tPetscObject) :: tPetscLimiter
      end type tPetscLimiter
      type, extends(tPetscObject) :: tPetscPartitioner
      end type tPetscPartitioner

      DM, parameter :: PETSC_NULL_DM = tDM(0)
      DMAdaptor, parameter :: PETSC_NULL_DMADAPTOR = tDMAdaptor(0)
      DMField, parameter :: PETSC_NULL_DMFIELD = tDMField(0)
      PetscQuadrature, parameter :: PETSC_NULL_QUADRATURE = tPetscQuadrature(0)
      PetscWeakForm, parameter :: PETSC_NULL_WEAKFORM = tPetscWeakForm(0)
      PetscDS, parameter :: PETSC_NULL_DS = tPetscDS(0)
      PetscFE, parameter :: PETSC_NULL_FE = tPetscFE(0)
      PetscSpace, parameter :: PETSC_NULL_SPACE = tPetscSpace(0)
      PetscDualSpace, parameter :: PETSC_NULL_DUALSPACE = tPetscDualSpace(0)
      PetscFV, parameter :: PETSC_NULL_FV = tPetscFV(0)
      PetscLimiter, parameter :: PETSC_NULL_LIMITER = tPetscLimiter(0)
      PetscPartitioner, parameter :: PETSC_NULL_PARTITIONER = tPetscPartitioner(0)
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
      PetscEnum, parameter :: DM_POLYTOPE_UNKNOWN_CELL = 15
      PetscEnum, parameter :: DM_POLYTOPE_UNKNOWN_FACE = 16
      PetscEnum, parameter :: DM_NUM_POLYTOPES = 17
!
! DMCopyLabelsMode
!
      PetscEnum, parameter :: DM_COPY_LABELS_REPLACE = 0
      PetscEnum, parameter :: DM_COPY_LABELS_KEEP    = 1
      PetscEnum, parameter :: DM_COPY_LABELS_FAIL    = 2
!
! DMReorderDefaultFlag
!
      PetscEnum, parameter :: DM_REORDER_DEFAULT_NOTSET = -1
      PetscEnum, parameter :: DM_REORDER_DEFAULT_FALSE = 0
      PetscEnum, parameter :: DM_REORDER_DEFAULT_TRUE = 1
!
!  PetscDTNodeType
!
      PetscEnum, parameter :: PETSCDTNODES_DEFAULT     = -1
      PetscEnum, parameter :: PETSCDTNODES_GAUSSJACOBI = 0
      PetscEnum, parameter :: PETSCDTNODES_EQUISPACED  = 1
      PetscEnum, parameter :: PETSCDTNODES_TANHSINH    = 2

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_DM
#endif
