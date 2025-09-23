!
!    Fortran kernel for gemv() BLAS operation. This version supports
!  matrix array stored in single precision but vectors in double
!
#include <petsc/finclude/petscsys.h>

pure subroutine MSGemv(bs, ncols, A, x, y)
  use, intrinsic :: ISO_C_binding
  implicit none(type, external)
  PetscInt, intent(in) :: bs, ncols
  MatScalar, intent(in) :: A(bs, ncols)
  PetscScalar, intent(in) :: x(ncols)
  PetscScalar, intent(out) :: y(bs)

  PetscInt :: i

  y(1:bs) = 0.0
  do i = 1, ncols
    y(1:bs) = y(1:bs) + A(1:bs, i)*x(i)
  end do
end subroutine MSGemv

pure subroutine MSGemvp(bs, ncols, A, x, y)
  use, intrinsic :: ISO_C_binding
  implicit none(type, external)
  PetscInt, intent(in) :: bs, ncols
  MatScalar, intent(in) :: A(bs, ncols)
  PetscScalar, intent(in) :: x(ncols)
  PetscScalar, intent(inout) :: y(bs)

  PetscInt :: i

  do i = 1, ncols
    y(1:bs) = y(1:bs) + A(1:bs, i)*x(i)
  end do
end subroutine MSGemvp

pure subroutine MSGemvm(bs, ncols, A, x, y)
  use, intrinsic :: ISO_C_binding
  implicit none(type, external)
  PetscInt, intent(in) :: bs, ncols
  MatScalar, intent(in) :: A(bs, ncols)
  PetscScalar, intent(in) :: x(ncols)
  PetscScalar, intent(inout) :: y(bs)

  PetscInt :: i

  do i = 1, ncols
    y(1:bs) = y(1:bs) - A(1:bs, i)*x(i)
  end do
end subroutine MSGemvm

pure subroutine MSGemvt(bs, ncols, A, x, y)
  use, intrinsic :: ISO_C_binding
  implicit none(type, external)
  PetscInt, intent(in) :: bs, ncols
  MatScalar, intent(in) :: A(bs, ncols)
  PetscScalar, intent(in) :: x(bs)
  PetscScalar, intent(inout) :: y(ncols)

  PetscInt :: i

  do i = 1, ncols
    y(i) = y(i) + sum(A(1:bs, i)*x(1:bs))
  end do
end subroutine MSGemvt

pure subroutine MSGemm(bs, A, B, C)
  use, intrinsic :: ISO_C_binding
  implicit none(type, external)
  PetscInt, intent(in) :: bs
  MatScalar, intent(in) :: B(bs, bs), C(bs, bs)
  MatScalar, intent(inout) :: A(bs, bs)

  PetscInt :: i, j

  do i = 1, bs
    do j = 1, bs
      A(i, j) = A(i, j) - sum(B(i, 1:bs)*C(1:bs, j))
    end do
  end do
end subroutine MSGemm

pure subroutine MSGemmi(bs, A, C, B)
  use, intrinsic :: ISO_C_binding
  implicit none(type, external)
  PetscInt, intent(in) :: bs
  MatScalar, intent(in) :: B(bs, bs), C(bs, bs)
  MatScalar, intent(out) :: A(bs, bs)

  PetscInt :: i, j

  do i = 1, bs
    do j = 1, bs
      A(i, j) = sum(B(i, 1:bs)*C(1:bs, j))
    end do
  end do
end subroutine MSGemmi
