! Introductory example that illustrates printing: Fortran Example



program main
#include <petsc/finclude/petscsys.h>
      use petscsys
      
      implicit none
      PetscErrorCode    :: ierr
      PetscMPIInt       :: myRank,mySize
      character(len=80) :: outputString
   
      ! Every PETSc routine should begin with the PetscInitialize() routine.

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr/= 0) then
        write(6,*) 'Unable to initialize PETSc'
        stop
      endif
      
      
      ! We can now change the communicator universe for PETSc
     
      call MPI_Comm_size(MPI_COMM_WORLD,mySize,ierr); CHKERRA(ierr)
      call MPI_Comm_rank(MPI_COMM_WORLD,myRank,ierr); CHKERRA(ierr)
     
      ! Here we would like to print only one message that represents all the processes in the group
      ! We use PetscPrintf() with the
      ! communicator PETSC_COMM_WORLD.  Thus, only one message is
      ! printed representng PETSC_COMM_WORLD, i.e., all the processors.
 
      write(outputString,*) 'No of Processors = ', mySize, ', rank = ',myRank,'\n'
      call PetscPrintf(PETSC_COMM_WORLD,outputString,ierr); CHKERRA(ierr)
     
      ! Here a barrier is used to separate the two program states.
      
      call MPI_Barrier(PETSC_COMM_WORLD,ierr); CHKERRA(ierr)

  
      ! Here we simply use PetscPrintf() with the communicator PETSC_COMM_SELF,
      ! where each process is considered separately and prints independently
      ! to the screen.  Thus, the output from different processes does not
      ! appear in any particular order.

      write(outputString,*) myRank,'Jumbled Hello World\n'
      call PetscPrintf(PETSC_COMM_SELF,outputString,ierr); CHKERRA(ierr)
      
      ! Always call PetscFinalize() before exiting a program.  This routine
      ! - finalizes the PETSc libraries as well as MPI
      ! - provides summary and diagnostic information if certain runtime
      !   options are chosen (e.g., -log_view).  See PetscFinalize()
      !  manpage for more information.

      call PetscFinalize(ierr)
      
end program main 
!/*TEST
!
!   test:
!
!TEST*/
