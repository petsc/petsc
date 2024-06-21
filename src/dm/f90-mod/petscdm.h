!
! Used by petscdmmod.F90 to create Fortran module file
!
#include "petsc/finclude/petscdm.h"

      type, extends(tPetscObject) :: tDM
      end type tDM
      DM, parameter :: PETSC_NULL_DM = tDM(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_DM
#endif

      type, extends(tPetscObject) :: tDMAdaptor
      end type tDMAdaptor
      DMAdaptor, parameter :: PETSC_NULL_DM_ADAPTOR = tDMAdaptor(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_DM_ADAPTOR
#endif

      type, extends(tPetscObject) :: tDMField
      end type tDMField
      DMField, parameter :: PETSC_NULL_DM_FIELD = tDMField(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_DM_FIELD
#endif

      type, extends(tPetscObject) :: tPetscQuadrature
      end type tPetscQuadrature
      PetscQuadrature, parameter :: PETSC_NULL_QUADRATURE = tPetscQuadrature(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_QUADRATURE
#endif

      type, extends(tPetscObject) :: tPetscWeakForm
      end type tPetscWeakForm
      PetscWeakForm, parameter :: PETSC_NULL_WEAKFORM = tPetscWeakForm(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_WEAKFORM
#endif

      type, extends(tPetscObject) :: tPetscDS
      end type tPetscDS
      PetscDS, parameter :: PETSC_NULL_DS = tPetscDS(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_DS
#endif

      type, extends(tPetscObject) :: tPetscFE
      end type tPetscFE
      PetscFE, parameter :: PETSC_NULL_FE = tPetscFE(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_FE
#endif

      type, extends(tPetscObject) :: tPetscSpace
      end type tPetscSpace
      PetscSpace, parameter :: PETSC_NULL_SPACE = tPetscSpace(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_SPACE
#endif

      type, extends(tPetscObject) :: tPetscDualSpace
      end type tPetscDualSpace
      PetscDualSpace, parameter :: PETSC_NULL_DUAL_SPACE = tPetscDualSpace(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_DUAL_SPACE
#endif

      type, extends(tPetscObject) :: tPetscFV
      end type tPetscFV
      PetscFV, parameter :: PETSC_NULL_FV = tPetscFV(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_FV
#endif

      type, extends(tPetscObject) :: tPetscLimiter
      end type tPetscLimiter
      PetscLimiter, parameter :: PETSC_NULL_LIMITER = tPetscLimiter(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_LIMITER
#endif

      type, extends(tPetscObject) :: tPetscPartitioner
      end type tPetscPartitioner
      PetscPartitioner, parameter :: PETSC_NULL_PARTITIONER = tPetscPartitioner(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_PARTITIONER
#endif
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
!
!  PetscGaussLobattoLegendreCreateType
!
      PetscEnum, parameter :: PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBR = 0
      PetscEnum, parameter :: PETSCGAUSSLOBATTOLEGENDRE_VIA_NEWTON = 1
