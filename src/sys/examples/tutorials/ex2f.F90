! Synchronized printing: Fortran Example


program main
#include <petsc/finclude/petscsys.h>
      use petscsys
      
      implicit none
      PetscErrorCode    :: ierr
      PetscMPIInt       :: myRank,mySize
      character(len=80) :: outputString
      
      ! Every PETSc program should begin with the PetscInitialize() routine.
      
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr /= 0) then
        write(6,*) 'Unable to initialize PETSc'
        stop
      endif
      
      ! The following MPI calls return the number of processes 
      ! being used and the rank of this process in the group
      
      call MPI_Comm_size(MPI_COMM_WORLD,mySize,ierr)
      CHKERRA(ierr)
      call MPI_Comm_rank(MPI_COMM_WORLD,myRank,ierr)
      CHKERRA(ierr)
      
      ! Here we would like to print only one message that represents
      ! all the processes in the group
      write(outputString,*) 'No of Processors = ', mysize, ', rank = ',myRank,'\n'
      call PetscPrintf(PETSC_COMM_WORLD,outputString,ierr)
      CHKERRA(ierr)
      
      write(outputString,*) myRank,'Synchronized Hello World\n'
      call PetscSynchronizedPrintf(PETSC_COMM_WORLD,outputString,ierr)
      CHKERRA(ierr)
      write(outputString,*) myRank,'Synchronized Hello World - Part II\n'
      call PetscSynchronizedPrintf(PETSC_COMM_WORLD,outputString,ierr)
      CHKERRA(ierr)
      call PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT,ierr)
      CHKERRA(ierr)
      
      ! Here a barrier is used to separate the two program states.
      call MPI_Barrier(PETSC_COMM_WORLD,ierr)
      CHKERRA(ierr)
      
      write(outputString,*) myRank,'Jumbled Hello World\n'
      call PetscPrintf(PETSC_COMM_SELF,outputString,ierr)
      CHKERRA(ierr)
      
      call PetscFinalize(ierr)
      CHKERRA(ierr)
      
end program main

!/*TEST
!
!   test:
!
!TEST*/
