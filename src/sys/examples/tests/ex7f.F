!
!
!     Formatted test for PetscSetCommWorld()
!
       program main
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
