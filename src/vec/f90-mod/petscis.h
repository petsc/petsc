!
!  Used by petscvecmod.F90 to create Fortran module file
!
#include "petsc/finclude/petscis.h"

      type, extends(tPetscObject) :: tIS
      end type tIS
      IS, parameter :: PETSC_NULL_IS = tIS(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_IS
#endif

      type, extends(tPetscObject) :: tISColoring
      end type tISColoring
      IS, parameter :: PETSC_NULL_IS_COLORING = tIS(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_IS_COLORING
#endif

      type, extends(tPetscObject) :: tPetscSection
      end type tPetscSection
      PetscSection, parameter :: PETSC_NULL_SECTION = tPetscSection(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_SECTION
#endif

      type, extends(tPetscObject) :: tPetscSectionSym
      end type tPetscSectionSym
      PetscSectionSym, parameter :: PETSC_NULL_SECTION_SYM = tPetscSectionSym(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_SECTION_SYM
#endif

      type, extends(tPetscObject) :: tPetscSF
      end type tPetscSF
      PetscSF, parameter :: PETSC_NULL_SF = tPetscSF(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_SF
#endif

      type :: tPetscLayout
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tPetscLayout
      PetscLayout, parameter :: PETSC_NULL_LAYOUT = tPetscLayout(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_LAYOUT
#endif

      type PetscSFNode
        PetscInt    rank
        PetscInt    index
      end type PetscSFNode

      PetscEnum, parameter :: IS_COLORING_GLOBAL = 0
      PetscEnum, parameter :: IS_COLORING_LOCAL = 1

      PetscEnum, parameter :: IS_GENERAL = 0
      PetscEnum, parameter :: IS_STRIDE = 1
      PetscEnum, parameter :: IS_BLOCK = 2

      PetscEnum, parameter :: IS_GTOLM_MASK =0
      PetscEnum, parameter :: IS_GTOLM_DROP = 1
!
!  ISInfo; must match those in include/petscis.h
!
      PetscEnum, parameter :: IS_INFO_MIN = -1
      PetscEnum, parameter :: IS_SORTED = 0
      PetscEnum, parameter :: IS_UNIQUE = 1
      PetscEnum, parameter :: IS_PERMUTATION = 2
      PetscEnum, parameter :: IS_INTERVAL = 3
      PetscEnum, parameter :: IS_IDENTITY = 4
      PetscEnum, parameter :: IS_INFO_MAX = 5
!
!  ISInfoType; must match those in include/petscis.h
!
      PetscEnum, parameter :: IS_LOCAL = 0
      PetscEnum, parameter :: IS_GLOBAL = 1

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::IS_COLORING_GLOBAL
!DEC$ ATTRIBUTES DLLEXPORT::IS_COLORING_LOCAL
!DEC$ ATTRIBUTES DLLEXPORT::IS_GENERAL
!DEC$ ATTRIBUTES DLLEXPORT::IS_STRIDE
!DEC$ ATTRIBUTES DLLEXPORT::IS_BLOCK
!DEC$ ATTRIBUTES DLLEXPORT::IS_GTOLM_MASK
!DEC$ ATTRIBUTES DLLEXPORT::IS_GTOLM_DROP
#endif
