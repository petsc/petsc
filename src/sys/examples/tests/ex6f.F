!    "$Id: ex6.F,v 1.12 2001/08/07 03:02:12 balay Exp $";
!
!  Progran to test random number generation routines from fortran.
!
       program main
#include "include/finclude/petsc.h"
#include "include/finclude/petscsys.h"
      integer         ierr
      PetscRandom     r
      PetscScalar     rand

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then 
         print*, 'Unable to begin PETSc program'
      endif

      call PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT,r,ierr)
      call PetscRandomGetValue(r,rand,ierr)
      print*, 'Random value:',rand

      call PetscRandomDestroy(r,ierr)
      call PetscFinalize(ierr)
      end
