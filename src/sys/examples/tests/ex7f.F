!
! "$Id: ex7f.F,v 1.4 1997/07/10 15:49:44 balay Exp balay $";
!
!
!     Formatted test for PetscSetCommWorld()
!

#include "FINCLUDE/petsc.h"

      integer size,ierr

      call MPI_Init(ierr)
      call PetscSetCommWorld(PETSC_COMM_SELF,ierr)
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
   
      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)
      if (size .ne. 1) then
        print*,'main:Error from PetscSetCommWorld()'
      endif

      call PetscFinalize(ierr)
      call MPI_Finalize(ierr)
      end
