!
!
!     Test for PetscFOpen() from Fortran
!
       program main
#include <finclude/petscsys.h>

      PetscErrorCode ierr
      PetscMPIInt rank
      PetscFortranAddr file
      character*100    joe
 
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)

      call PetscFOpen(PETSC_COMM_WORLD,'testfile','w',file,ierr)
      
      call PetscFPrintf(PETSC_COMM_WORLD,file,'Hi once \n',ierr)
      call PetscSynchronizedFPrintf(PETSC_COMM_WORLD,file,'Hi twice \n',     &
     &                              ierr)
      call PetscSynchronizedFlush(PETSC_COMM_WORLD,ierr)

      write (FMT=*,UNIT=joe) 'greetings from ',rank,'\n'
      call PetscSynchronizedFPrintf(PETSC_COMM_WORLD,file,joe,ierr)
      call PetscSynchronizedFlush(PETSC_COMM_WORLD,ierr)

      call PetscFClose(PETSC_COMM_WORLD,file,ierr)

      call PetscFinalize(ierr)
      end
