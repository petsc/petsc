!
!   This program tests MatCreateVecs() for Shell Matrix
!
#include <petsc/finclude/petscmat.h>
module ex120fmodule
  use petscmat
  implicit none

contains
  subroutine mymatgetvecs(A, x, y, ierr)

    PetscErrorCode ierr
    Mat A
    Vec x, y
    PetscInt tw

    tw = 12
    PetscCallA(VecCreateSeq(PETSC_COMM_SELF, tw, x, ierr))
    PetscCallA(VecCreateSeq(PETSC_COMM_SELF, tw, y, ierr))
  end
end module ex120fmodule

program main
  use petscmat
  use ex120fmodule
  implicit none

  PetscErrorCode ierr
  Vec x, y
  Mat m
  PetscInt tw

  PetscCallA(PetscInitialize(ierr))

  tw = 12
  PetscCallA(MatCreateShell(PETSC_COMM_SELF, tw, tw, tw, tw, 0, m, ierr))
  PetscCallA(MatAssemblyBegin(m, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd(m, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatShellSetOperation(m, MATOP_CREATE_VECS, mymatgetvecs, ierr))
  PetscCallA(MatCreateVecs(m, x, y, ierr))
  PetscCallA(MatDestroy(m, ierr))
  PetscCallA(VecDestroy(x, ierr))
  PetscCallA(VecDestroy(y, ierr))
  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!   test:
!      nsize: 2
!      output_file: output/empty.out
!
!TEST*/
