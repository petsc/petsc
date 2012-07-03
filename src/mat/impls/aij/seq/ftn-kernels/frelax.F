!
!    Fortran kernels for SOR relaxations
!
#include <finclude/petscsysdef.h>
!
      subroutine FortranRelaxAIJForwardZero(n,omega,x,ai,aj,            &
     &           adiag,idiag,aa,b)
      implicit none
      PetscScalar x(0:*),aa(0:*)
      PetscScalar b(0:*),idiag(0:*)
      PetscReal   omega
      PetscInt    n,ai(0:*)
      PetscInt    aj(0:*),adiag(0:*)

      PetscInt    i,j,jstart,jend
      PetscScalar sum
!     
!     Forward Solve with zero initial guess
!
      x(0) = omega*b(0)*idiag(0)
      do 20 i=1,n-1
         jstart = ai(i)
         jend   = adiag(i) - 1
         sum    = b(i)
         do 30 j=jstart,jend
            sum  = sum -  aa(j) * x(aj(j))
 30      continue
         x(i) = omega*sum*idiag(i)
 20   continue
      
!     return
      end
!
!-------------------------------------------------------------------
!      
      subroutine FortranRelaxAIJBackwardZero(n,omega,x,ai,aj,           &
     &                                  adiag,idiag,aa,b)
      implicit none
      PetscScalar x(0:*),aa(0:*)
      PetscScalar b(0:*),idiag(0:*)
      PetscReal   omega
      PetscInt    aj(0:*),adiag(0:*)
      PetscInt    n,ai(0:*)

      PetscInt    i,j,jstart,jend
      PetscScalar sum
!
!     Backward solve with zero initial guess
!
      do 40 i=n-1,0,-1
         jstart  = adiag(i) + 1
         jend    = ai(i+1) - 1
         sum     = b(i)
         do 50 j=jstart,jend
            sum = sum - aa(j)* x(aj(j)) 
 50      continue
         x(i)    = omega*sum*idiag(i)
 40   continue
      return
      end
      
!-------------------------------------------------------------------
!      
      subroutine FortranRelaxAIJForward(n,omega,x,ai,aj,adiag,aa,b)
      implicit none
      PetscScalar x(0:*),aa(0:*),b(0:*)
      PetscReal   omega
      PetscInt    n,ai(0:*)
      PetscInt    aj(0:*),adiag(0:*)

      PetscInt    i,j,jstart,jend
      PetscScalar sum
!
!     Forward solve with non-zero initial guess
!
      do 40 i=0,n-1
         sum    = b(i)

         jstart = ai(i)
         jend    = ai(i+1) - 1
         do 50 j=jstart,jend
            sum = sum - aa(j)* x(aj(j)) 
 50      continue
         x(i)    = (1.0 - omega)*x(i) +                                 &
     &              omega*(sum + aa(adiag(i))*x(i))/ aa(adiag(i))
 40   continue
      return
      end
      
!-------------------------------------------------------------------
!      
      subroutine FortranRelaxAIJBackward(n,omega,x,ai,aj,adiag,aa,b)
      implicit none
      PetscScalar x(0:*),aa(0:*),b(0:*)
      PetscReal   omega
      PetscInt    n,ai(0:*)
      PetscInt    aj(0:*),adiag(0:*)

      PetscInt    i,j,jstart,jend
      PetscScalar sum
!
!     Backward solve with non-zero initial guess
!
      do 40 i=n-1,0,-1
         sum    = b(i)

         jstart = ai(i)
         jend   = ai(i+1) - 1
         do 50 j=jstart,jend
            sum = sum - aa(j)* x(aj(j)) 
 50      continue
         x(i)    = (1.0 - omega)*x(i) +                                 &
     &              omega*(sum + aa(adiag(i))*x(i)) / aa(adiag(i))
 40   continue
      return
      end
      
