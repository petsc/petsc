! Synchronized printing: Fortran Example

program main
#include <petsc/finclude/petscsys.h>
      use petscmpi  ! or mpi or mpi_f08
      use petscsys

      implicit none
      PetscErrorCode                    :: ierr
      PetscMPIInt                       :: rank,size
      character(len=PETSC_MAX_PATH_LEN) :: outputString

      ! Every PETSc program should begin with the PetscInitialize() routine.

      PetscCallA(PetscInitialize(ierr))

      ! The following MPI calls return the number of processes
      ! being used and the rank of this process in the group

      PetscCallMPIA(MPI_Comm_size(MPI_COMM_WORLD,size,ierr))
      PetscCallMPIA(MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr))

      ! Here we would like to print only one message that represents
      ! all the processes in the group
      write(outputString,*) 'No of Processors = ', size, ', rank = ',rank,'\n'
      PetscCallA(PetscPrintf(PETSC_COMM_WORLD,outputString,ierr))

      write(outputString,*) rank,'Synchronized Hello World\n'
      PetscCallA(PetscSynchronizedPrintf(PETSC_COMM_WORLD,outputString,ierr))

      write(outputString,*) rank,'Synchronized Hello World - Part II\n'
      PetscCallA(PetscSynchronizedPrintf(PETSC_COMM_WORLD,outputString,ierr))
      PetscCallA(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT,ierr))

      ! Here a barrier is used to separate the two program states.
      PetscCallMPIA(MPI_Barrier(PETSC_COMM_WORLD,ierr))

      write(outputString,*) rank,'Jumbled Hello World\n'
      PetscCallA(PetscPrintf(PETSC_COMM_SELF,outputString,ierr))

      PetscCallA(PetscFinalize(ierr))
end program main

!/*TEST
!
!   test:
!
!TEST*/
