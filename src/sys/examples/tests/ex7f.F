!
! "$Id: ex7f.F,v 1.5 1998/03/24 16:42:22 balay Exp balay $";
!
!
!     Formatted test for PetscSetCommWorld()
!

#include "finclude/petsc.h"

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
