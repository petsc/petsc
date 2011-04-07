!
!    Test AO with on IS with 0 entries - Fortran version of ex4.c
!
      program main
      implicit none
#include <finclude/petscsys.h>
#include <finclude/petscao.h>

      PetscErrorCode ierr
      AO             ao
      PetscInt       localvert(4),nlocal
      PetscMPIInt    rank

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)

      nlocal = 0
      if (rank .eq. 0) then
         nlocal = 4
         localvert(1) = 0
         localvert(2) = 1
         localvert(3) = 2
         localvert(4) = 3
      endif

!     Test AOCreateBasic() 
      call AOCreateBasic(PETSC_COMM_WORLD, nlocal, localvert,             &
     &     PETSC_NULL_INTEGER,ao,ierr)
      call AODestroy(ao,ierr)

!     Test AOCreateMemoryScalable() 
      call AOCreateMemoryScalable(PETSC_COMM_WORLD, nlocal, localvert,    &
     &     PETSC_NULL_INTEGER,ao,ierr)
      call AODestroy(ao,ierr)

      call PetscFinalize(ierr)
      end

