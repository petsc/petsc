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

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
         print*, 'Unable to begin PETSc program'
      endif

      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)
!      if (size .ne. 2) SETERRA(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,'Two ranks only')
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      call PetscSubcommCreate(PETSC_COMM_WORLD,r,ierr)
      call PetscSubcommSetFromOptions(r,ierr)
      call PetscSubcommSetTypeGeneral(r,rank,rank,ierr)

      call PetscSubcommGetChild(r,scomm,ierr)
      call PetscSubcommView(r,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call PetscSubcommDestroy(r,ierr)
      call PetscFinalize(ierr)
      end

!
!/*TEST
!
!   test:
!     nsize: 2
!
!TEST*/
