C
C  "$Id: fmultadd.F,v 1.1 1997/09/18 01:23:50 balay Exp balay $";
C
C    Fortran kernel for sparse matrix vector product in AIJ format
C
C
      subroutine FortranSolveAIJ(n,x,ai,aj,adiag,aa,b)
      implicit none
#include "include/FINCLUDE/petsc.h"

      Scalar           x(0:*),aa(0:*),b(0:*)
      integer          n, ai(0:*),aj(0:*),adiag(0:*)

      integer          i,j,ai_i,nz,adiag_i
      Scalar           sum
C     
C     Forward Solve
C
      x(0) = b(0)
      do 20 i=1,n-1
         ai_i = ai(i)
         nz   = adiag(i) - ai_i
         sum  = b(i)
         do 30 j=1,nz
            sum  = sum - aa(ai_i) * x(aj(ai_i))
            ai_i = ai_i + 1
 30      continue
      x(i) = sum
 20   continue
      
C
C     Backward solve the upper triangular
C
      do 40 i=n-1,0,-1
         adiag_i = adiag(i)
         nz      = ai(i+1) - adiag_i - 1
         sum     = x(i)
         do 50 j=1,nz
            adiag_i = adiag_i + 1
            sum = sum - aa(adiag_i) * x(aj(adiag_i))
 50      continue
         x(i)    = sum * aa(adiag(i))
 40   continue
      return
      end
      
