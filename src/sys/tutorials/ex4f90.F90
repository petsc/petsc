!
!     This introductory example illustrates running PETSc on a subset
!     of processes
!
! -----------------------------------------------------------------------

      program main
#include <petsc/finclude/petscsys.h>
      use petscmpi  ! or mpi or mpi_f08
      use petscsys
      implicit none

      PetscErrorCode ierr
      PetscMPIInt rank, size, zero, two

!     We must call MPI_Init() first, making us, not PETSc, responsible
!     for MPI

      PetscCallMPIA(MPI_Init(ierr))

!     We can now change the communicator universe for PETSc

      zero = 0
      two = 2
      PetscCallMPIA(MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr))
      PetscCallMPIA(MPI_Comm_split(MPI_COMM_WORLD,mod(rank,two),zero,PETSC_COMM_WORLD,ierr))

!     Every PETSc routine should begin with the PetscInitialize()
!     routine.

      PetscCallA(PetscInitialize(PETSC_NULL_CHARACTER,ierr))

!     The following MPI calls return the number of processes being used
!     and the rank of this process in the group.

      PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD,size,ierr))
      PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))

!     Here we would like to print only one message that represents all
!     the processes in the group.
      if (rank .eq. 0) write(6,100) size,rank
 100  format('No of Procs = ',i4,' rank = ',i4)

!     Always call PetscFinalize() before exiting a program.  This
!     routine - finalizes the PETSc libraries as well as MPI - provides
!     summary and diagnostic information if certain runtime options are
!     chosen (e.g., -log_view).  See PetscFinalize() manpage for more
!     information.

      PetscCallA(PetscFinalize(ierr))
      PetscCallMPIA(MPI_Comm_free(PETSC_COMM_WORLD,ierr))

!     Since we initialized MPI, we must call MPI_Finalize()

      PetscCallMPIA( MPI_Finalize(ierr))
      end

!
!/*TEST
!
!   test:
!
!TEST*/
