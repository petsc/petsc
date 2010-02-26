

!
!      Fortran main program that uses Sieve to read in and
!    partition a cell-centered unstructured mesh

      program ex12f90
#include "finclude/petsc.h"     
      DM mesh
      PetscErrorCode   ierr
      Vec gvec,lvec

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call LoadMesh(mesh,ierr)
      call DMCreateGlobalVector(mesh,gvec,ierr)
      call DMCreateLocalVector(mesh,lvec,ierr)
      call VecDestroy(gvec,ierr)
      call VecDestroy(lvec,ierr)
      call PetscFinalize(ierr)
      end


