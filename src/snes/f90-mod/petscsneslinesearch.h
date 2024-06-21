!
! Used by petscsnesmod.F90 to create Fortran module file
!

      type, extends(tPetscObject) :: tSNESLineSearch
      end type tSNESLineSearch
      SNESLineSearch, parameter :: PETSC_NULL_SNES_LINESEARCH = tSNESLineSearch(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_SNES_LINESEARCH
#endif
!
!     SNESLineSearchReason
!
      PetscEnum, parameter :: SNES_LINESEARCH_SUCCEEDED       = 0
      PetscEnum, parameter :: SNES_LINESEARCH_FAILED_NANORINF = 1
      PetscEnum, parameter :: SNES_LINESEARCH_FAILED_DOMAIN   = 2
      PetscEnum, parameter :: SNES_LINESEARCH_FAILED_REDUCT   = 3
      PetscEnum, parameter :: SNES_LINESEARCH_FAILED_USER     = 4
      PetscEnum, parameter :: SNES_LINESEARCH_FAILED_FUNCTION = 5
