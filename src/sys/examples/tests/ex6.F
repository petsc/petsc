C    "$Id: ex5.F,v 1.7 1996/03/20 03:28:28 bsmith Exp $";
C
C  Simple PETSc Program written in Fortran 77.
C
#include "include/FINCLUDE/petsc.h"
      integer         ierr
      PetscRandom     r
      Scalar          rand

      call PetscInitialize(PETSC_NULL_CHAR,ierr)
      if (ierr .ne. 0) then 
         print*, 'Unable to begin PETSc program'
      endif

      call PetscRandomCreate(MPI_COMM_WORLD,RANDOM_DEFAULT,r,ierr);
      call PetscRandomGetValue(r,rand,ierr)
      
      print*, 'Random value:', rank

      call PetscRandomDestroy(r,ierr)
      call PetscFinalize(ierr)
      end
