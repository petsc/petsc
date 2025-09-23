!
!
!    Fortran kernel for the MDot() vector routine
!
#include <petsc/finclude/petscsys.h>
!
pure subroutine FortranMDot4(x, y1, y2, y3, y4, n, sum1, sum2, sum3, sum4)
  use, intrinsic :: ISO_C_binding
  implicit none(type, external)
  PetscScalar, intent(inout) :: sum1, sum2, sum3, sum4
  PetscScalar, intent(in) :: x(*), y1(*), y2(*), y3(*), y4(*)
  PetscInt, intent(in) :: n

  PetscInt :: i

  PETSC_AssertAlignx(16, x(1))
  PETSC_AssertAlignx(16, y1(1))
  PETSC_AssertAlignx(16, y2(1))
  PETSC_AssertAlignx(16, y3(1))
  PETSC_AssertAlignx(16, y4(1))

  do i = 1, n
    sum1 = sum1 + x(i)*PetscConj(y1(i))
    sum2 = sum2 + x(i)*PetscConj(y2(i))
    sum3 = sum3 + x(i)*PetscConj(y3(i))
    sum4 = sum4 + x(i)*PetscConj(y4(i))
  end do
end subroutine FortranMDot4

pure subroutine FortranMDot3(x, y1, y2, y3, n, sum1, sum2, sum3)
  use, intrinsic :: ISO_C_binding
  implicit none(type, external)
  PetscScalar, intent(inout) :: sum1, sum2, sum3
  PetscScalar, intent(in) :: x(*), y1(*), y2(*), y3(*)
  PetscInt, intent(in) :: n

  PetscInt :: i

  PETSC_AssertAlignx(16, x(1))
  PETSC_AssertAlignx(16, y1(1))
  PETSC_AssertAlignx(16, y2(1))
  PETSC_AssertAlignx(16, y3(1))

  do i = 1, n
    sum1 = sum1 + x(i)*PetscConj(y1(i))
    sum2 = sum2 + x(i)*PetscConj(y2(i))
    sum3 = sum3 + x(i)*PetscConj(y3(i))
  end do
end subroutine FortranMDot3

pure subroutine FortranMDot2(x, y1, y2, n, sum1, sum2)
  use, intrinsic :: ISO_C_binding
  implicit none(type, external)
  PetscScalar, intent(inout) :: sum1, sum2
  PetscScalar, intent(in) :: x(*), y1(*), y2(*)
  PetscInt, intent(in) :: n

  PetscInt :: i

  PETSC_AssertAlignx(16, x(1))
  PETSC_AssertAlignx(16, y1(1))
  PETSC_AssertAlignx(16, y2(1))

  do i = 1, n
    sum1 = sum1 + x(i)*PetscConj(y1(i))
    sum2 = sum2 + x(i)*PetscConj(y2(i))
  end do
end subroutine FortranMDot2

pure subroutine FortranMDot1(x, y1, n, sum1)
  use, intrinsic :: ISO_C_binding
  implicit none(type, external)
  PetscScalar, intent(inout) :: sum1
  PetscScalar, intent(in) :: x(*), y1(*)
  PetscInt, intent(in) :: n

  PetscInt :: i

  PETSC_AssertAlignx(16, x(1))
  PETSC_AssertAlignx(16, y1(1))

  do i = 1, n
    sum1 = sum1 + x(i)*PetscConj(y1(i))
  end do

end subroutine FortranMDot1
