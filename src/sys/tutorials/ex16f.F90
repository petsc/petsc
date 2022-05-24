! Tests calling PetscOptionsSetValue() before PetscInitialize(): Fortran Example

program main
#include <petsc/finclude/petscsys.h>
      use petscmpi  ! or mpi or mpi_f08
      use petscsys

      implicit none
      PetscErrorCode :: ierr
      PetscMPIInt  ::  rank,size
      character(len=80) :: outputString

      ! Every PETSc routine should begin with the PetscInitialize() routine.

      PetscCallA(PetscOptionsSetValue(PETSC_NULL_OPTIONS,"-no_signal_handler","true",ierr))
      PetscCallA(PetscInitialize(ierr))

      ! Since when PetscInitialize() returns with an error the PETSc data structures
      ! may not be set up hence we cannot call CHKERRA() hence directly return the error code.

      ! Since PetscOptionsSetValue() is called before the PetscInitialize() we cannot call
      ! CHKERRA() on the error code and just return it directly.

      ! We can now change the communicator universe for PETSc

      PetscCallMPIA(MPI_Comm_size(MPI_COMM_WORLD,size,ierr))
      PetscCallMPIA(MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr))
      write(outputString,*) 'Number of processors =',size,'rank =',rank,'\n'
      PetscCallA(PetscPrintf(PETSC_COMM_WORLD,outputString,ierr))
      PetscCallA(PetscFinalize(ierr))
end program main

!/*TEST
!
!   test:
!      requires: defined(PETSC_USE_LOG)
!      nsize: 2
!      args: -options_view -get_total_flops
!      filter: egrep -v "(malloc|display|nox|Total flops|saws_port_auto_select|vecscatter_mpi1|options_left|error_output_stdout|check_pointer_intensity|cuda_initialize|use_gpu_aware_mpi|checkstack)"
!
!TEST*/
