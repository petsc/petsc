!
!
!    Fortran kernel for the WAXPY() vector routine
!
#include "include/finclude/petscdef.h"
!
      subroutine FortranWAXPY(n,a,x,y,w)
      implicit none
      PetscScalar  a
      PetscScalar  x(*),y(*),w(*)
      PetscInt n

      PetscInt i

      do 10,i=1,n
        w(i) = a*x(i) + y(i)
 10   continue

      return 
      end
