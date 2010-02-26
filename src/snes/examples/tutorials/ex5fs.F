!
! ---------------------------------------------------------------------
!
!  Fortran version of the user function based on shared memory
!  this routine is called only by MPI process 0 in the computation
!  but uses threads to run the loops in parallel.

!
!  Input Parameter:
!  x - global array containing input values
!
!  Output Parameters:
!  f - global array containing output values
!  ierr - error code 
!
!  Notes:
!  This routine uses standard Fortran-style computations over a 2-dim array.
!
      subroutine ApplicationFunctionFortran(lambda,mx,my,x,f,ierr)

      implicit none

#include "finclude/petscsys.h"
      integer  ierr,mx,my

!  Input/output variables:
      PetscScalar   x(mx,my),f(mx,my),lambda


!  Local variables:
      PetscScalar   two,one,hx,hy,hxdhy,hydhx,sc
      PetscScalar   u,uxx,uyy
      integer  i,j

      one    = 1.0
      two    = 2.0
      hx     = one/dble(mx-1)
      hy     = one/dble(my-1)
      sc     = hx*hy*lambda
      hxdhy  = hx/hy
      hydhx  = hy/hx

!  Compute function over the entire grid

      do 20 j=1,my
         do 10 i=1,mx
            if (i .eq. 1 .or. j .eq. 1                                  &
     &             .or. i .eq. mx .or. j .eq. my) then
               f(i,j) = x(i,j)
            else
               u = x(i,j) 
               uxx = hydhx * (two*u                                     &
     &                - x(i-1,j) - x(i+1,j))
               uyy = hxdhy * (two*u - x(i,j-1) - x(i,j+1))
               f(i,j) = uxx + uyy - sc*exp(u)
            endif
 10      continue
 20   continue

      return
      end

