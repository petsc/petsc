!
!  "$Id: fsolvebaij.F,v 1.2 1998/10/07 20:07:24 bsmith Exp bsmith $";
!
!    Fortran kernel for sparse triangular solve in the BAIJ matrix format
! This ONLY works for factorizations in the NATURAL ORDERING, i.e.
! with MatSolve_SeqBAIJ_4_NaturalOrdering()
!
      subroutine FortranSolveBAIJ4(n,x,ai,aj,adiag,a,b)
      implicit none
#include "include/finclude/petscdef.h"

      Scalar           x(0:*),a(0:*),b(0:*)
      integer          n, ai(0:*),aj(0:*),adiag(0:*)

      integer          i,j,jstart,jend,idx,adx,jdx
      Scalar           s1, s2, s3, s4
      Scalar           x1,x2,x3,x4
!     
!     Forward Solve
!
      x(0) = b(0)
      x(1) = b(1)
      x(2) = b(2)
      x(3) = b(3)
      idx  = 0
      do 20 i=1,n-1
         jstart = ai(i)
         jend   = adiag(i) - 1
         adx    = 16*jstart
         idx    = idx + 4
         s1     = b(idx)
         s2     = b(idx+1)
         s3     = b(idx+2)
         s4     = b(idx+3)
         do 30 j=jstart,jend
           jdx   = 4*aj(j)
            
           x1    = x(jdx)
           x2    = x(jdx+1)
           x3    = x(jdx+2)
           x4    = x(jdx+3)
           s1 = s1-(a(adx)*x1  +a(adx+4)*x2+a(adx+8)*x3 +a(adx+12)*x4)
           s2 = s2-(a(adx+1)*x1+a(adx+5)*x2+a(adx+9)*x3 +a(adx+13)*x4)
           s3 = s3-(a(adx+2)*x1+a(adx+6)*x2+a(adx+10)*x3+a(adx+14)*x4)
           s4 = s4-(a(adx+3)*x1+a(adx+7)*x2+a(adx+11)*x3+a(adx+15)*x4)
           adx = adx + 16
 30      continue
         x(idx)   = s1
         x(idx+1) = s2
         x(idx+2) = s3
         x(idx+3) = s4
 20   continue
      
!
!     Backward solve the upper triangular
!
      do 40 i=n-1,0,-1
         jstart  = adiag(i) + 1
         jend    = ai(i+1) - 1
         adx     = 16*jstart
         s1      = x(idx)
         s2      = x(idx+1)
         s3      = x(idx+2)
         s4      = x(idx+3)
         do 50 j=jstart,jend
           jdx   = 4*aj(j)
           x1    = x(jdx)
           x2    = x(jdx+1)
           x3    = x(jdx+2)
           x4    = x(jdx+3)
           s1 = s1-(a(adx)*x1  +a(adx+4)*x2+a(adx+8)*x3 +a(adx+12)*x4)
           s2 = s2-(a(adx+1)*x1+a(adx+5)*x2+a(adx+9)*x3 +a(adx+13)*x4)
           s3 = s3-(a(adx+2)*x1+a(adx+6)*x2+a(adx+10)*x3+a(adx+14)*x4)
           s4 = s4-(a(adx+3)*x1+a(adx+7)*x2+a(adx+11)*x3+a(adx+15)*x4)
           adx = adx + 16
 50      continue
         adx      = 16*adiag(i)
         x(idx)   = a(adx)*s1  +a(adx+4)*s2+a(adx+8)*s3 +a(adx+12)*s4
         x(idx+1) = a(adx+1)*s1+a(adx+5)*s2+a(adx+9)*s3 +a(adx+13)*s4
         x(idx+2) = a(adx+2)*s1+a(adx+6)*s2+a(adx+10)*s3+a(adx+14)*s4
         x(idx+3) = a(adx+3)*s1+a(adx+7)*s2+a(adx+11)*s3+a(adx+15)*s4
         idx      = idx - 4
 40   continue
      return
      end
      
