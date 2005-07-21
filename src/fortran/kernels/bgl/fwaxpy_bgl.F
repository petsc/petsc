!
!
!    Fortran kernel for the WAXPY() vector routine
!
#include "include/finclude/petscdef.h"
!
      subroutine FortranWAXPY_bgl(n,a,x,y,w)
      implicit none
      PetscScalar  a
      PetscScalar  x(*),y(*),w(*)
      PetscInt n

      PetscInt i

      call ALIGNX(16,x(1))
      call ALIGNX(16,y(1))
      call ALIGNX(16,w(1))

      do 10,i=1,n
        w(i) = a*x(i) + y(i)
 10   continue

      return 
      end
