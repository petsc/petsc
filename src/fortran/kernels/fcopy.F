!
!
!    Fortran kernel for the copy vector routine
!
#include "include/finclude/petscdef.h"
!
      subroutine FortranCopy(n,x,y)
      implicit none
      PetscScalar  x(*),y(*)
      integer n

      integer i

      do 10,i=1,n
        y(i) = x(i)
 10   continue

      return 
      end


      subroutine FortranZero(n,x)
      implicit none
      PetscScalar  x(*)
      integer n

      integer i

      do 10,i=1,n
        x(i) = 0.0
 10   continue

      return 
      end

