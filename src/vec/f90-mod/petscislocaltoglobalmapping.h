!
!  Used by petscvecmod.F90 to create Fortran module file
!
      type, extends(tPetscObject) :: tISLocalToGlobalMapping
      end type tISLocalToGlobalMapping
      ISLocalToGlobalMapping, parameter :: PETSC_NULL_IS_LOCALTOGLOBALMAPPING = tISLocalToGlobalMapping(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_IS_LOCALTOGLOBALMAPPING
#endif
