C
C "$Id: ex7f.F,v 1.1 1996/09/23 23:22:12 bsmith Exp bsmith $";
C
C
C     Formatted test for PetscSetCommWorld()
C

#include "FINCLUDE/petsc.h"

      integer size,ierr

      call MPI_Init(ierr)
      call PetscSetCommWorld(MPI_COMM_SELF,ierr)
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
   
      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)
      if (size .ne. 1) then
        print*,'main:Error from PetscSetCommWorld()'
      endif

      call PetscFinalize(ierr)
      call MPI_Finalize(ierr)
      stop
      end
