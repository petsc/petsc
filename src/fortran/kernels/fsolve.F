C
C  "$Id: fsolve.F,v 1.5 1997/12/07 04:04:29 bsmith Exp bsmith $";
C
C    Fortran kernel for sparse triangular solve in the AIJ matrix format
C
C
      subroutine FortranSolveAIJ(n,x,ai,aj,adiag,aa,b)
      implicit none
#include "include/FINCLUDE/petsc.h"

      Scalar           x(0:*),aa(0:*),b(0:*)
      integer          n, ai(0:*),aj(0:*),adiag(0:*)

      integer          i,j,jstart,jend
      Scalar           sum
C     
C     Forward Solve
C
      x(0) = b(0)
      do 20 i=1,n-1
         jstart = ai(i)
         jend   = adiag(i) - 1
         sum    = b(i)
         do 30 j=jstart,jend
            sum  = sum -  aa(j) * x(aj(j))
 30      continue
         x(i) = sum
 20   continue
      
C
C     Backward solve the upper triangular
C
      do 40 i=n-1,0,-1
         jstart  = adiag(i) + 1
         jend    = ai(i+1) - 1
         sum     = x(i)
         do 50 j=jstart,jend
            sum = sum - aa(j)* x(aj(j)) 
 50      continue
         x(i)    = sum * aa(adiag(i))
 40   continue
      return
      end
      
