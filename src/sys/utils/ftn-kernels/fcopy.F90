!
!
!    Fortran kernel for the copy vector routine
!
#include <petsc/finclude/petscsys.h>
!
pure subroutine FortranCopy(n,x,y)
  implicit none (type, external)
  PetscScalar, intent(in) :: x(*)
  PetscScalar, intent(inout) :: y(*)
  PetscInt, intent(in) :: n

  PetscInt i

  PETSC_AssertAlignx(16,x(1))
  PETSC_AssertAlignx(16,y(1))

  do i=1,n
    y(i) = x(i)
  end do
end subroutine FortranCopy

pure subroutine FortranZero(n,x)
  implicit none (type, external)
  PetscScalar, intent(inout) :: x(*)
  PetscInt, intent(in) :: n

  PetscInt i

  PETSC_AssertAlignx(16,x(1))

  do i=1,n
    x(i) = 0.0
  end do
end subroutine FortranZero
