!
!  Progran to test random number generation routines from fortran.
!
       program main
#include "finclude/petscsys.h"
      PetscErrorCode  ierr
      PetscRandom     r
      PetscScalar     rand

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then 
         print*, 'Unable to begin PETSc program'
      endif

      call PetscRandomCreate(PETSC_COMM_WORLD,r,ierr)
      call PetscRandomSetFromOptions(r,ierr)
      call PetscRandomGetValue(r,rand,ierr)
      print*, 'Random value:',rand

      call PetscRandomDestroy(r,ierr)
      call PetscFinalize(ierr)
      end
