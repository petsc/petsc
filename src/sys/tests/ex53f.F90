!
!  Program to test PetscSubcomm.
!
      program main

#include <petsc/finclude/petscsys.h>
      use petscsys
      implicit none

      PetscErrorCode  ierr
      PetscSubcomm    r
      PetscMPIInt     rank,size
      MPI_Comm        scomm

      PetscCallA(PetscInitialize(ierr))
      PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD,size,ierr))
!      if (size .ne. 2) SETERRA(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,'Two ranks only')
      PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))
      PetscCallA(PetscSubcommCreate(PETSC_COMM_WORLD,r,ierr))
      PetscCallA(PetscSubcommSetFromOptions(r,ierr))
      PetscCallA(PetscSubcommSetTypeGeneral(r,rank,rank,ierr))

      PetscCallA(PetscSubcommGetChild(r,scomm,ierr))
      PetscCallA(PetscSubcommView(r,PETSC_VIEWER_STDOUT_WORLD,ierr))
      PetscCallA(PetscSubcommDestroy(r,ierr))
      PetscCallA(PetscFinalize(ierr))
      end

!
!/*TEST
!
!   test:
!     nsize: 2
!
!TEST*/
