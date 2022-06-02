!
!  Simple PETSc Program written in Fortran
!
       program main
#include <petsc/finclude/petscsys.h>
       use petscmpi  ! or mpi or mpi_f08
       use petscsys
       implicit none

       PetscErrorCode  ierr
       PetscInt f(1)
       PetscCallA(PetscInitialize(ierr))
       f(1) = 1
       PetscCallMPIA(MPI_Allreduce(MPI_IN_PLACE,f,1,MPIU_INTEGER,MPI_MIN,PETSC_COMM_WORLD,ierr))
       PetscCallA(PetscFinalize(ierr))
       end

!/*TEST
!
!   test:
!
!TEST*/
