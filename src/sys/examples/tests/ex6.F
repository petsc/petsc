!    "$Id: ex6.F,v 1.6 1997/09/22 15:18:58 balay Exp balay $";
!
!  Progran to test random number generation routines from fortran.
!
#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/sys.h"
      integer         ierr
      PetscRandom     r
      Scalar          rand

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then 
         print*, 'Unable to begin PETSc program'
      endif

      call PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT,r,ierr)
      call PetscRandomGetValue(r,rand,ierr)
      print*, 'Random value:', rand

      call PetscRandomDestroy(r,ierr)
      call PetscFinalize(ierr)
      end
