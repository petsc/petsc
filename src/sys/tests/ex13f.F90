!
!  Program to test object composition from Fortran
!
      program main

#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscviewer.h>
      use petscsys
      implicit none

      PetscErrorCode                 ierr
      PetscObject                    o1, o2, o3
      character*(PETSC_MAX_PATH_LEN) name

      PetscCallA(PetscInitialize(ierr))
      PetscCallA(PetscViewerCreate(PETSC_COMM_WORLD,o1,ierr))
      PetscCallA(PetscViewerCreate(PETSC_COMM_WORLD,o2,ierr))
      name = 'matt'
      PetscCallA(PetscObjectCompose(o1,name,o2,ierr))
      PetscCallA(PetscObjectQuery(o1,name,o3,ierr))
      if (o2 .ne. o3) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'PetscObjectQuery failed'); endif

      PetscCallA(PetscViewerDestroy(o1,ierr))
      PetscCallA(PetscViewerDestroy(o2,ierr))
      PetscCallA(PetscFinalize(ierr))
      end

!
!/*TEST
!
!   test:
!      suffix: 0
!
!TEST*/
