!
!    Fortran kernel for gemv() BLAS operation. This version supports
!  matrix array stored in single precision but vectors in double
!
#include <petsc/finclude/petscsys.h>

subroutine MSGemv(bs,ncols,A,x,y)
  implicit none
  PetscInt          bs,ncols
  MatScalar        A(bs,ncols)
  PetscScalar      x(ncols),y(bs)

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

subroutine MSGemvp(bs,ncols,A,x,y)
  implicit none
  PetscInt          bs,ncols
  MatScalar        A(bs,ncols)
  PetscScalar      x(ncols),y(bs)

  PetscInt         i, j

  do i=1,ncols
    do j=1,bs
      y(j) = y(j) + A(j,i)*x(i)
    end do
  end do

end subroutine MSGemvp

subroutine MSGemvm(bs,ncols,A,x,y)
  implicit none
  PetscInt          bs,ncols
  MatScalar        A(bs,ncols)
  PetscScalar      x(ncols),y(bs)

  PetscInt         i, j

  do i=1,ncols
    do j=1,bs
      y(j) = y(j) - A(j,i)*x(i)
    end do
  end do

end subroutine MSGemvm

subroutine MSGemvt(bs,ncols,A,x,y)
  implicit none
  PetscInt          bs,ncols
  MatScalar        A(bs,ncols)
  PetscScalar      x(bs),y(ncols)

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

subroutine MSGemm(bs,A,B,C)
  implicit none
  PetscInt    bs
  MatScalar   A(bs,bs),B(bs,bs),C(bs,bs)
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

subroutine MSGemmi(bs,A,C,B)
  implicit none
  PetscInt    bs
  MatScalar   A(bs,bs),B(bs,bs),C(bs,bs)
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
