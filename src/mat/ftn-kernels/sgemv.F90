!
!    Fortran kernel for gemv() BLAS operation. This version supports
!  matrix array stored in single precision but vectors in double
!
#include <petsc/finclude/petscsys.h>

pure subroutine MSGemv(bs,ncols,A,x,y)
  implicit none (type, external)
  PetscInt, intent(in) :: bs,ncols
  MatScalar, intent(in) :: A(bs,ncols)
  PetscScalar, intent(in) :: x(ncols)
  PetscScalar, intent(out) :: y(bs)

  PetscInt         i,j

  do j=1,bs
    y(j) = 0.0d0
  end do

  do i=1,ncols
    do j=1,bs
      y(j) = y(j) + A(j,i)*x(i)
    end do
  end do
end subroutine MSGemv

pure subroutine MSGemvp(bs,ncols,A,x,y)
  implicit none (type, external)
  PetscInt, intent(in) :: bs,ncols
  MatScalar, intent(in) :: A(bs,ncols)
  PetscScalar, intent(in) :: x(ncols)
  PetscScalar, intent(inout) :: y(bs)

  PetscInt         i, j

  do i=1,ncols
    do j=1,bs
      y(j) = y(j) + A(j,i)*x(i)
    end do
  end do
end subroutine MSGemvp

pure subroutine MSGemvm(bs,ncols,A,x,y)
  implicit none (type, external)
  PetscInt, intent(in) :: bs,ncols
  MatScalar, intent(in) :: A(bs,ncols)
  PetscScalar, intent(in) :: x(ncols)
  PetscScalar, intent(inout) :: y(bs)

  PetscInt         i, j

  do i=1,ncols
    do j=1,bs
      y(j) = y(j) - A(j,i)*x(i)
    end do
  end do
end subroutine MSGemvm

pure subroutine MSGemvt(bs,ncols,A,x,y)
  implicit none (type, external)
  PetscInt, intent(in) :: bs,ncols
  MatScalar, intent(in) :: A(bs,ncols)
  PetscScalar, intent(in) :: x(bs)
  PetscScalar, intent(inout) :: y(ncols)

  PetscInt          i,j
  PetscScalar      sum

  do  i=1,ncols
    sum = y(i)
    do  j=1,bs
      sum = sum + A(j,i)*x(j)
    end do
    y(i) = sum
  end do
end subroutine MSGemvt

pure subroutine MSGemm(bs,A,B,C)
  implicit none (type, external)
  PetscInt, intent(in) :: bs
  MatScalar, intent(in) :: B(bs,bs),C(bs,bs)
  MatScalar, intent(inout) :: A(bs,bs)

  PetscScalar sum
  PetscInt    i,j,k

  do i=1,bs
    do j=1,bs
      sum = A(i,j)
      do k=1,bs
        sum = sum - B(i,k)*C(k,j)
      end do
      A(i,j) = sum
    end do
  end do
end subroutine MSGemm

pure subroutine MSGemmi(bs,A,C,B)
  implicit none (type, external)
  PetscInt, intent(in) :: bs
  MatScalar, intent(in) :: B(bs,bs),C(bs,bs)
  MatScalar, intent(out) :: A(bs,bs)

  PetscScalar sum
  PetscInt    i,j,k

  do i=1,bs
    do j=1,bs
      sum = 0.0d0
      do  k=1,bs
        sum = sum + B(i,k)*C(k,j)
      end do
      A(i,j) = sum
    end do
  end do
end subroutine MSGemmi
