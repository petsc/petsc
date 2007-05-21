#define PETSC_AVOID_DECLARATIONS
#include "include/finclude/petscall.h"

!
!      Fortran main program that uses Sieve to read in and
!    partition a cell-centered unstructured mesh

      program ex12f90
      use mex12f90


      PetscErrorCode   ierr
      type(appctx)     app


      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call handlemesh(ierr)
      call PetscFinalize(ierr)
      end


