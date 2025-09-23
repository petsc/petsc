!
!  Program to test object composition from Fortran
!
program main

#include <petsc/finclude/petscsys.h>
  use petscsys
  implicit none

  PetscErrorCode ierr
  PetscViewer o1, o2, o3
  character*(4) name
  PetscCopyMode :: mode = PETSC_COPY_VALUES

  PetscCallA(PetscInitialize(ierr))
  PetscCallA(PetscViewerASCIIOpen(PETSC_COMM_WORLD, "stdout", o1, ierr))
  PetscCallA(PetscViewerASCIIOpen(PETSC_COMM_WORLD, "stderr", o2, ierr))
  name = 'matt'
  PetscCallA(PetscObjectCompose(o1, name, o2, ierr))
  PetscCallA(PetscObjectQuery(o1, name, o3, ierr))
  PetscCheckA(o2 == o3, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'PetscObjectQuery failed')

  if (mode == PETSC_COPY_VALUES) then
    PetscCallA(PetscViewerDestroy(o1, ierr))
  end if
  PetscCallA(PetscViewerDestroy(o2, ierr))
  PetscCallA(PetscFinalize(ierr))
end

!
!/*TEST
!
!   build:
!     requires: defined(PETSC_HAVE_FORTRAN_TYPE_STAR)
!
!   test:
!     suffix: 0
!     output_file: output/empty.out
!
!TEST*/
