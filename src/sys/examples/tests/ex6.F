!    "$Id: ex6.F,v 1.8 1998/04/15 18:00:56 balay Exp bsmith $";
!
!  Progran to test random number generation routines from fortran.
!
#include "include/finclude/petsc.h"
#include "include/finclude/sys.h"
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
