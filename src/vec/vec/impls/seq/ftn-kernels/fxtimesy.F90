#include <petsc/finclude/petscsys.h>
!

      subroutine Fortranxtimesy(x,y,z,n)
      implicit none
      PetscScalar  x(*),y(*),z(*)
      PetscInt n
      PetscInt i
      PETSC_AssertAlignx(16,x(1))
      PETSC_AssertAlignx(16,y(1))
      PETSC_AssertAlignx(16,z(1))
      do 10,i=1,n
         z(i) = x(i) * y(i)
  10  continue
      return
      end

