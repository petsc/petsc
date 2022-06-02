! Introductory example that illustrates printing: Fortran Example

program main
#include <petsc/finclude/petscsys.h>
      use petscsys

      implicit none
      PetscErrorCode    :: ierr
      PetscMPIInt       :: rank,size
      character(len=80) :: outputString

      ! Every PETSc routine should begin with the PetscInitialize() routine.
      PetscCallA(PetscInitialize(ierr))

      ! We can now change the communicator universe for PETSc
      PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD,size,ierr))
      PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))

      ! Here we would like to print only one message that represents all the processes in the group
      ! We use PetscPrintf() with the
      ! communicator PETSC_COMM_WORLD.  Thus, only one message is
      ! printed representng PETSC_COMM_WORLD, i.e., all the processors.

      write(outputString,*) 'No of Processors = ', size, ', rank = ',rank,'\n'
      PetscCallA(PetscPrintf(PETSC_COMM_WORLD,outputString,ierr))

      ! Here a barrier is used to separate the two program states.
      PetscCallMPIA(MPI_Barrier(PETSC_COMM_WORLD,ierr))

      ! Here we simply use PetscPrintf() with the communicator PETSC_COMM_SELF,
      ! where each process is considered separately and prints independently
      ! to the screen.  Thus, the output from different processes does not
      ! appear in any particular order.

      write(outputString,*) rank,'Jumbled Hello World\n'
      PetscCallA(PetscPrintf(PETSC_COMM_SELF,outputString,ierr))

      ! Always call PetscFinalize() before exiting a program.  This routine
      ! - finalizes the PETSc libraries as well as MPI
      ! - provides summary and diagnostic information if certain runtime
      !   options are chosen (e.g., -log_view).  See PetscFinalize()
      !  manpage for more information.

      PetscCallA(PetscFinalize(ierr))

end program main
!/*TEST
!
!   test:
!
!TEST*/
