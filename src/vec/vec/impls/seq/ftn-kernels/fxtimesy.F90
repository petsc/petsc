#include <petsc/finclude/petscsys.h>
!

pure subroutine Fortranxtimesy(x,y,z,n)
  implicit none (type, external)
  PetscScalar, intent(in) :: x(*),y(*)
  PetscScalar, intent(inout) :: z(*)
  PetscInt, intent(in) :: n

  PetscInt i

  PETSC_AssertAlignx(16,x(1))
  PETSC_AssertAlignx(16,y(1))
  PETSC_AssertAlignx(16,z(1))

  do i=1,n
    z(i) = x(i) * y(i)
  end do
end subroutine Fortranxtimesy
