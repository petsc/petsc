! Tests calling PetscOptionsSetValue() before PetscInitialize(): Fortran Example

program main
#include <petsc/finclude/petscsys.h>
      use petscsys
      
      implicit none
      PetscErrorCode :: ierr
      PetscMPIInt  ::  myRank,mySize
      character(len=80) :: outputString  
  
      ! Every PETSc routine should begin with the PetscInitialize() routine.
     
      call PetscOptionsSetValue(PETSC_NULL_OPTIONS,"-no_signal_handler","true",ierr)
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr/=0) then
        write(6,*) 'Unable to initialize PETSc'
        stop
      endif
      
      ! Since when PetscInitialize() returns with an error the PETSc data structures
      ! may not be set up hence we cannot call CHKERRA() hence directly return the error code.

      ! Since PetscOptionsSetValue() is called before the PetscInitialize() we cannot call 
      ! CHKERRA() on the error code and just return it directly.
      
      ! We can now change the communicator universe for PETSc

      call MPI_Comm_size(MPI_COMM_WORLD,mySize,ierr); CHKERRA(ierr)
      call MPI_Comm_rank(MPI_COMM_WORLD,myRank,ierr); CHKERRA(ierr)
      write(outputString,*) 'Number of processors =',mySize,'rank =',myRank,'\n'
      call PetscPrintf(PETSC_COMM_WORLD,outputString,ierr); CHKERRA(ierr)
      call PetscFinalize(ierr)

end program main 

!/*TEST
!
!   test:
!      nsize: 2
!      args: -options_view -get_total_flops
!      filter: egrep -v "(malloc|display|nox|Total flops|saws_port_auto_select|vecscatter_mpi1|options_left|error_output_stdout|check_pointer_intensity|cuda_initialize|use_gpu_aware_mpi)"
!
!TEST*/
