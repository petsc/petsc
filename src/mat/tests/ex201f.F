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
      call MatCreateShell(PETSC_COMM_SELF,i12,i12,i12,i12,i0,B,ierr)
      call MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY, ierr)

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

      call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif

      viewer = PETSC_VIEWER_STDOUT_SELF
      i12 = 12
      i0 = 0
      call VecCreateSeq(PETSC_COMM_SELF, i12, x, ierr)
      call VecCreateSeq(PETSC_COMM_SELF, i12, y, ierr)
      call VecCreateSeq(PETSC_COMM_SELF, i12, z, ierr)
      call MatCreateShell(PETSC_COMM_SELF,i12,i12,i12,i12,i0,m,ierr)
      call MatShellSetManageScalingShifts(m,ierr)
      call MatAssemblyBegin(m, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(m, MAT_FINAL_ASSEMBLY, ierr)

      op = MATOP_MULT
      call MatShellSetOperation(m, op, mymatmult, ierr)
      op = MATOP_MULT_ADD
      call MatShellSetOperation(m, op, mymatmultadd, ierr)
      op = MATOP_MULT_TRANSPOSE
      call MatShellSetOperation(m, op, mymatmulttranspose, ierr)
      op = MATOP_MULT_TRANSPOSE_ADD
      call MatShellSetOperation(m, op, mymatmulttransposeadd, ierr)
      op = MATOP_TRANSPOSE
      call MatShellSetOperation(m, op, mymattranspose, ierr)
      op = MATOP_GET_DIAGONAL
      call MatShellSetOperation(m, op, mymatgetdiagonal, ierr)
      op = MATOP_DIAGONAL_SCALE
      call MatShellSetOperation(m, op, mymatdiagonalscale, ierr)
      op = MATOP_ZERO_ENTRIES
      call MatShellSetOperation(m, op, mymatzeroentries, ierr)
      op = MATOP_AXPY
      call MatShellSetOperation(m, op, mymataxpy, ierr)
      op = MATOP_SHIFT
      call MatShellSetOperation(m, op, mymatshift, ierr)
      op = MATOP_DIAGONAL_SET
      call MatShellSetOperation(m, op, mymatdiagonalset, ierr)
      op = MATOP_DESTROY
      call MatShellSetOperation(m, op, mymatdestroy, ierr)
      op = MATOP_VIEW
      call MatShellSetOperation(m, op, mymatview, ierr)
      op = MATOP_CREATE_VECS
      call MatShellSetOperation(m, op, mymatgetvecs, ierr)

      call MatMult(m, x, y, ierr)
      call MatMultAdd(m, x, y, z, ierr)
      call MatMultTranspose(m, x, y, ierr)
      call MatMultTransposeAdd(m, x, y, z, ierr)
      call MatTranspose(m, MAT_INITIAL_MATRIX, mt, ierr)
      call MatGetDiagonal(m, x, ierr)
      call MatDiagonalScale(m, x, y, ierr)
      call MatZeroEntries(m, ierr)
      a = 102.
      call MatAXPY(m, a, mt, SAME_NONZERO_PATTERN, ierr)
      call MatShift(m, a, ierr)
      call MatDiagonalSet(m, x, INSERT_VALUES, ierr)
      call MatView(m, viewer, ierr)
      call MatCreateVecs(m, x, y, ierr)
      call MatDestroy(m,ierr)
      call MatDestroy(mt, ierr)
      call VecDestroy(x, ierr)
      call VecDestroy(y, ierr)
      call VecDestroy(z, ierr)

      call PetscFinalize(ierr)
      end

!/*TEST
!
!   test:
!     args: -malloc_dump
!     filter: sort -b
!     filter_output: sort -b
!
!TEST*/
