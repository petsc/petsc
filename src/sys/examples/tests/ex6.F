!    "$Id: ex6.F,v 1.10 2000/05/05 22:14:28 balay Exp bsmith $";
!
!  Progran to test random number generation routines from fortran.
!
       program main
#include "include/finclude/petsc.h"
#include "include/finclude/petscsys.h"
      integer         ierr
      PetscRandom     r
      Scalar          rand

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
