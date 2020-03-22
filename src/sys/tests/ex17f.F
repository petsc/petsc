!
!
!     Test for PetscFOpen() from Fortran
!
       program main
#include <petsc/finclude/petscsys.h>
       use petscsys
       implicit none

      PetscErrorCode ierr
      PetscMPIInt rank
      PetscFortranAddr file
      character*100    joe

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
         print*,'Unable to initialize PETSc'
         stop
      endif
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)

      call PetscFOpen(PETSC_COMM_WORLD,'testfile','w',file,ierr)

      call PetscFPrintf(PETSC_COMM_WORLD,file,'Hi once \n',ierr)
      call PetscSynchronizedFPrintf(PETSC_COMM_WORLD,file,'Hi twice \n',     &
     &                              ierr)
      call PetscSynchronizedFlush(PETSC_COMM_WORLD,file,ierr)

      write (FMT=*,UNIT=joe) 'greetings from ',rank,'\n'
      call PetscSynchronizedFPrintf(PETSC_COMM_WORLD,file,joe,ierr)
      call PetscSynchronizedFlush(PETSC_COMM_WORLD,file,ierr)

      call PetscFClose(PETSC_COMM_WORLD,file,ierr)

      call PetscSynchronizedPrintf(PETSC_COMM_WORLD,'Hi twice \n',          &
     &                              ierr)
      call PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT,ierr)

      call PetscFinalize(ierr)
      end

!
!/*TEST
!
!   test:
!      nsize: 3
!
!TEST*/
