#define dim 3
#define n   7

      subroutine product(A,v,w)
      implicit none
      double precision A(n,n,dim),v(n,n,n),w(n,n,n),sum
      integer i,k,p,q,r
      

      
      k = 1
      do p=1,n
        do q=1,n
          do r=1,n
             sum = 0.0d0
             do i=1,n
               sum = sum + A(r,i,k)*v(i,q,p)
             enddo
             w(r,q,p) = sum
          enddo
        enddo
      enddo

      k = 2
      do p=1,n
        do q=1,n
          do r=1,n
             sum = 0.0d0
             do i=1,n
               sum = sum + A(q,i,k)*v(r,i,p)
             enddo
             w(r,q,p) = sum
          enddo
        enddo
      enddo

      k = 3
      do p=1,n
        do q=1,n
          do r=1,n
             sum = 0.0d0
             do i=1,n
               sum = sum + A(p,i,k)*v(r,p,i)
             enddo
             w(r,q,p) = sum
          enddo
        enddo
      enddo

      return
      end

      program main
      implicit none
#include "finclude/petsc.h"      
#include "finclude/petsclog.h"      
      double precision A(n,n,dim),v(n,n,n),w(n,n,n)
      double precision starttime,endtime
      integer          ierr
      external product

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call PetscGetTime(starttime,ierr)
      call product(A,v,w)      
      call PetscGetTime(endtime,ierr)

      print*,'Flop rate ',3*2.0d-6*n**(dim+1)/(endtime-starttime)
      call PetscFinalize(ierr)

      stop
      end

