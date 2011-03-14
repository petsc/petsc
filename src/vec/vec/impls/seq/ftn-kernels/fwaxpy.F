!
!
!    Fortran kernel for the WAXPY() vector routine
!
#include <finclude/petscsysdef.h>
!
      subroutine FortranWAXPY(n,a,x,y,w)
      implicit none
      PetscScalar  a
      PetscScalar  x(*),y(*),w(*)
      PetscInt n

      PetscInt i

      PETSC_AssertAlignx(16,x(1))
      PETSC_AssertAlignx(16,y(1))
      PETSC_AssertAlignx(16,w(1))

      do 10,i=1,n
        w(i) = a*x(i) + y(i)
 10   continue

      return 
      end
