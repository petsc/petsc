!
!
!   This program demonstrates use of MatShellSetOperation()
!
      subroutine mymatmult(A, x, y, ierr)
#include <petsc/finclude/petscmat.h>
      use petscmat
      implicit none

      Mat A
      Vec x, y
      PetscErrorCode ierr

      print*, "Called MatMult"
      return
      end

      subroutine mymatmultadd(A, x, y, z, ierr)
      use petscmat
      implicit none
      Mat A
      Vec x, y, z
      PetscErrorCode ierr

      print*, "Called MatMultAdd"
      return
      end

      subroutine mymatmulttranspose(A, x, y, ierr)
      use petscmat
      implicit none
      Mat A
      Vec x, y
      PetscErrorCode ierr

      print*, "Called MatMultTranspose"
      return
      end

      subroutine mymatmulttransposeadd(A, x, y, z, ierr)
      use petscmat
      implicit none
      Mat A
      Vec x, y, z
      PetscErrorCode ierr

      print*, "Called MatMultTransposeAdd"
      return
      end

      subroutine mymattranspose(A, reuse, B, ierr)
      use petscmat
      implicit none
      Mat A, B
      MatReuse reuse
      PetscErrorCode ierr
      PetscInt i12,i0

      i12 = 12
      i0 = 0
      PetscCallA(MatCreateShell(PETSC_COMM_SELF,i12,i12,i12,i12,i0,B,ierr))
      PetscCallA(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY, ierr))
      PetscCallA(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY, ierr))

      print*, "Called MatTranspose"
      return
      end

      subroutine mymatgetdiagonal(A, x, ierr)
      use petscmat
      implicit none
      Mat A
      Vec x
      PetscErrorCode ierr

      print*, "Called MatGetDiagonal"
      return
      end

      subroutine mymatdiagonalscale(A, x, y, ierr)
      use petscmat
      implicit none
      Mat A
      Vec x, y
      PetscErrorCode ierr

      print*, "Called MatDiagonalScale"
      return
      end

      subroutine mymatzeroentries(A, ierr)
      use petscmat
      implicit none
      Mat A
      PetscErrorCode ierr

      print*, "Called MatZeroEntries"
      return
      end

      subroutine mymataxpy(A, alpha, B, str, ierr)
      use petscmat
      implicit none
      Mat A, B
      PetscScalar alpha
      MatStructure str
      PetscErrorCode ierr

      print*, "Called MatAXPY"
      return
      end

      subroutine mymatshift(A, alpha, ierr)
      use petscmat
      implicit none
      Mat A
      PetscScalar alpha
      PetscErrorCode ierr

      print*, "Called MatShift"
      return
      end

      subroutine mymatdiagonalset(A, x, ins, ierr)
      use petscmat
      implicit none
      Mat A
      Vec x
      InsertMode ins
      PetscErrorCode ierr

      print*, "Called MatDiagonalSet"
      return
      end

      subroutine mymatdestroy(A, ierr)
      use petscmat
      implicit none
      Mat A
      PetscErrorCode ierr

      print*, "Called MatDestroy"
      return
      end

      subroutine mymatview(A, viewer, ierr)
      use petscmat
      implicit none
      Mat A
      PetscViewer viewer
      PetscErrorCode ierr

      print*, "Called MatView"
      return
      end

      subroutine mymatgetvecs(A, x, y, ierr)
      use petscmat
      implicit none
      Mat A
      Vec x, y
      PetscErrorCode ierr

      print*, "Called MatCreateVecs"
      return
      end

      program main
      use petscmat
      implicit none

      Mat     m, mt
      Vec     x, y, z
      PetscScalar a
      PetscViewer viewer
      MatOperation op
      PetscErrorCode ierr
      PetscInt i12,i0
      external mymatmult
      external mymatmultadd
      external mymatmulttranspose
      external mymatmulttransposeadd
      external mymattranspose
      external mymatgetdiagonal
      external mymatdiagonalscale
      external mymatzeroentries
      external mymataxpy
      external mymatshift
      external mymatdiagonalset
      external mymatdestroy
      external mymatview
      external mymatgetvecs

      PetscCallA(PetscInitialize(ierr))

      viewer = PETSC_VIEWER_STDOUT_SELF
      i12 = 12
      i0 = 0
      PetscCallA(VecCreateSeq(PETSC_COMM_SELF, i12, x, ierr))
      PetscCallA(VecCreateSeq(PETSC_COMM_SELF, i12, y, ierr))
      PetscCallA(VecCreateSeq(PETSC_COMM_SELF, i12, z, ierr))
      PetscCallA(MatCreateShell(PETSC_COMM_SELF,i12,i12,i12,i12,i0,m,ierr))
      PetscCallA(MatShellSetManageScalingShifts(m,ierr))
      PetscCallA(MatAssemblyBegin(m, MAT_FINAL_ASSEMBLY, ierr))
      PetscCallA(MatAssemblyEnd(m, MAT_FINAL_ASSEMBLY, ierr))

      op = MATOP_MULT
      PetscCallA(MatShellSetOperation(m, op, mymatmult, ierr))
      op = MATOP_MULT_ADD
      PetscCallA(MatShellSetOperation(m, op, mymatmultadd, ierr))
      op = MATOP_MULT_TRANSPOSE
      PetscCallA(MatShellSetOperation(m, op, mymatmulttranspose, ierr))
      op = MATOP_MULT_TRANSPOSE_ADD
      PetscCallA(MatShellSetOperation(m, op, mymatmulttransposeadd, ierr))
      op = MATOP_TRANSPOSE
      PetscCallA(MatShellSetOperation(m, op, mymattranspose, ierr))
      op = MATOP_GET_DIAGONAL
      PetscCallA(MatShellSetOperation(m, op, mymatgetdiagonal, ierr))
      op = MATOP_DIAGONAL_SCALE
      PetscCallA(MatShellSetOperation(m, op, mymatdiagonalscale, ierr))
      op = MATOP_ZERO_ENTRIES
      PetscCallA(MatShellSetOperation(m, op, mymatzeroentries, ierr))
      op = MATOP_AXPY
      PetscCallA(MatShellSetOperation(m, op, mymataxpy, ierr))
      op = MATOP_SHIFT
      PetscCallA(MatShellSetOperation(m, op, mymatshift, ierr))
      op = MATOP_DIAGONAL_SET
      PetscCallA(MatShellSetOperation(m, op, mymatdiagonalset, ierr))
      op = MATOP_DESTROY
      PetscCallA(MatShellSetOperation(m, op, mymatdestroy, ierr))
      op = MATOP_VIEW
      PetscCallA(MatShellSetOperation(m, op, mymatview, ierr))
      op = MATOP_CREATE_VECS
      PetscCallA(MatShellSetOperation(m, op, mymatgetvecs, ierr))

      PetscCallA(MatMult(m, x, y, ierr))
      PetscCallA(MatMultAdd(m, x, y, z, ierr))
      PetscCallA(MatMultTranspose(m, x, y, ierr))
      PetscCallA(MatMultTransposeAdd(m, x, y, z, ierr))
      PetscCallA(MatTranspose(m, MAT_INITIAL_MATRIX, mt, ierr))
      PetscCallA(MatGetDiagonal(m, x, ierr))
      PetscCallA(MatDiagonalScale(m, x, y, ierr))
      PetscCallA(MatZeroEntries(m, ierr))
      a = 102.
      PetscCallA(MatAXPY(m, a, mt, SAME_NONZERO_PATTERN, ierr))
      PetscCallA(MatShift(m, a, ierr))
      PetscCallA(MatDiagonalSet(m, x, INSERT_VALUES, ierr))
      PetscCallA(MatView(m, viewer, ierr))
      PetscCallA(MatCreateVecs(m, x, y, ierr))
      PetscCallA(MatDestroy(m,ierr))
      PetscCallA(MatDestroy(mt, ierr))
      PetscCallA(VecDestroy(x, ierr))
      PetscCallA(VecDestroy(y, ierr))
      PetscCallA(VecDestroy(z, ierr))

      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!   test:
!     args: -malloc_dump
!     filter: sort -b
!     filter_output: sort -b
!
!TEST*/
