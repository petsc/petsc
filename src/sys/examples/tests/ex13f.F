!
!
!     Tests PetscSetCommWorld()
!
#include "finclude/petsc.h"

      integer   ierr,rank,size,color
      MPI_Comm  newcomm

      call MPI_Init(ierr)
      call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
      call MPI_Comm_size(MPI_COMM_WORLD,size,ierr)

!
!       make two new communicators each half the size of original
!
      if (2*rank .lt. size) then 
        color = 0
      else 
        color = 1
      endif
      call MPI_Comm_split(MPI_COMM_WORLD,color,0,newcomm,ierr)

      call PetscSetCommWorld(newcomm,ierr)

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      write(6,100) rank
  100 format('rank = ',i3)

      call PetscFinalize(ierr)
      call MPI_Finalize(ierr)
      end
