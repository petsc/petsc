!
!  "$Id: fmaxpy.F,v 1.10 2001/08/07 03:05:24 balay Exp $";
!
!    Fortran kernel for the AYPX() vector routine
!
#include "include/finclude/petscdef.h"
!
      subroutine FortranAYPX(n,a,x,y)
      implicit none
      PetscScalar  a
      PetscScalar  x(*),y(*)
      integer n

      integer i

      do 10,i=1,n
        y(i) = x(i) + a*y(i)
 10   continue

      return 
      end
