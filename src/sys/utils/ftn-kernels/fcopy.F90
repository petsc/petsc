!
!
!    Fortran kernel for the copy vector routine
!
#include <petsc/finclude/petscsys.h>
!
subroutine FortranCopy(n,x,y)
  implicit none
  PetscScalar  x(*),y(*)
  PetscInt n
  PetscInt i

  PETSC_AssertAlignx(16,x(1))
  PETSC_AssertAlignx(16,y(1))

  do i=1,n
    y(i) = x(i)
  end do

end subroutine FortranCopy

subroutine FortranZero(n,x)
  implicit none
  PetscScalar  x(*)
  PetscInt n
  PetscInt i

  PETSC_AssertAlignx(16,x(1))

  do i=1,n
    x(i) = 0.0
  end do

end subroutine FortranZero
