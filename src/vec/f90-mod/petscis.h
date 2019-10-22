!
!
!  Include file for Fortran use of the IS (index set) package in PETSc
!
#include "petsc/finclude/petscis.h"

      type tIS
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tIS
      type tISColoring
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tISColoring
      type tPetscSection
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tPetscSection
      type tPetscSectionSym
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tPetscSectionSym
      type tPetscSF
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tPetscSF
      type PetscSFNode
        sequence
        PetscInt    rank
        PetscInt    index
      end type PetscSFNode

      IS, parameter :: PETSC_NULL_IS = tIS(0)
      PetscSF, parameter :: PETSC_NULL_SF = tPetscSF(0)
      PetscSection, parameter :: PETSC_NULL_SECTION =                   &
     & tPetscSection(0)
      PetscSectionSym, parameter :: PETSC_NULL_SECTIONSYM =             &
     & tPetscSectionSym(0)

      PetscEnum IS_COLORING_GLOBAL
      PetscEnum IS_COLORING_LOCAL
      parameter (IS_COLORING_GLOBAL = 0,IS_COLORING_LOCAL = 1)

      PetscEnum IS_GENERAL
      PetscEnum IS_STRIDE
      PetscEnum IS_BLOCK
      parameter (IS_GENERAL = 0,IS_STRIDE = 1,IS_BLOCK = 2)

      PetscEnum IS_GTOLM_MASK
      PetscEnum IS_GTOLM_DROP
      parameter (IS_GTOLM_MASK =0,IS_GTOLM_DROP = 1)

!
!  ISInfo; must match those in include/petscis.h
!
      PetscEnum IS_INFO_MIN
      PetscEnum IS_SORTED
      PetscEnum IS_UNIQUE
      PetscEnum IS_PERMUTATION
      PetscEnum IS_INTERVAL
      PetscEnum IS_IDENTITY
      PetscEnum IS_INFO_MAX

      parameter(IS_INFO_MIN = -1)
      parameter(IS_SORTED = 0)
      parameter(IS_UNIQUE = 1)
      parameter(IS_PERMUTATION = 2)
      parameter(IS_INTERVAL = 3)
      parameter(IS_IDENTITY = 4)
      parameter(IS_INFO_MAX = 5)

!
!  ISInfoType; must match those in include/petscis.h
!
      PetscEnum IS_LOCAL
      PetscEnum IS_GLOBAL
      parameter(IS_LOCAL = 0)
      parameter(IS_GLOBAL = 1)

!
!  End of Fortran include file for the IS package in PETSc

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_IS
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_SF
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_SECTION
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_SECTIONSYM
!DEC$ ATTRIBUTES DLLEXPORT::IS_COLORING_GLOBAL
!DEC$ ATTRIBUTES DLLEXPORT::IS_COLORING_LOCAL
!DEC$ ATTRIBUTES DLLEXPORT::IS_GENERAL
!DEC$ ATTRIBUTES DLLEXPORT::IS_STRIDE
!DEC$ ATTRIBUTES DLLEXPORT::IS_BLOCK
!DEC$ ATTRIBUTES DLLEXPORT::IS_GTOLM_MASK
!DEC$ ATTRIBUTES DLLEXPORT::IS_GTOLM_DROP
#endif
