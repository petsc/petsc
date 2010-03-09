#define dim 3
#define n 7
#define ntest 1

#undef __FUNCT__
#define __FUNCT__ "main"
      program main
      implicit none
#include "finclude/petscsys.h"      
#include "finclude/petsclog.h"      
      double precision A(n,n,dim),B(n,n,dim),v(n,n,n),w(n,n,n),sum
      double precision starttime,endtime
      integer          ierr,i,j,k
!      external product3D
!      external product3DMXM

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      do i = 1,dim
         do j = 1,n
            do k = 1,n
               A(k,j,i) = 2.0d0*i*(j+k)
               B(j,k,i) = A(k,j,i)
            enddo
         enddo
      enddo
      do i = 1,n
         do j = 1,n
            do k = 1,n
               v(k,j,i) = 3.0d0*(i*j)/k
               w(k,j,i) = 0.0d0
            enddo
         enddo
      enddo
      print *, 'n is ',n 
      call product3D(A,B,v,w)      
      sum = 0.0d0
      do i = 1,n
         do j = 1,n
            do k = 1,n
               sum = sum + w(k,j,i)*w(k,j,i)
               w(k,j,i) = 0.0d0
            enddo
         enddo
      enddo
      print *, 'Norm of w is ',dsqrt(sum)
!      call PetscGetTime(starttime,ierr)
      call product3DNEW(A,B,v,w)      
!      call PetscGetTime(endtime,ierr)

!      print*,'Total Flop rate ',3*2.0d-6*n**(dim+1)/(endtime-starttime)
      sum = 0.0d0
      do i = 1,n
         do j = 1,n
            do k = 1,n
               sum = sum + w(k,j,i)*w(k,j,i)
            enddo
         enddo
      enddo
      print *, 'Norm of w is ',dsqrt(sum)
      call PetscFinalize(ierr)

      stop
      end

#undef __FUNCT__
#define __FUNCT__ "product3D"
      subroutine product3D(A,B,v,w)
      implicit none
      double precision A(n,n,dim),B(n,n,dim),v(n,n,n),w(n,n,n),sum
      double precision st1,st2,st3,et1,et2,et3,elap1,elap2,elap3
      double precision flops,flops1,flops2,flops3
      integer i,j,k,p,q,r,ierr,itest
      

! First do I x I x A  
      elap1 = 0.0d0    
      do itest = 1,ntest
         call PetscGetTime(st1,ierr)
         do p=1,n
            do q=1,n
               do r=1,n
                  sum = 0.0d0
                  do i=1,n
                     sum = sum + A(r,i,1)*v(i,q,p)
                  enddo
                  w(r,q,p) = sum
               enddo
            enddo
         enddo
         call PetscGetTime(et1,ierr)
         elap1 = elap1 + (et1-st1)
      enddo
      elap1 = elap1/ntest
! Next do I x A x I
      elap2 = 0.0d0    
      do itest = 1,ntest
         call PetscGetTime(st2,ierr)
         do p=1,n
            do q=1,n
               do r=1,n
                  sum = 0.0d0
                  do i=1,n
                     sum = sum + A(q,i,2)*v(r,i,p)
                  enddo
                  w(r,q,p) = w(r,q,p)+sum
               enddo
            enddo
         enddo
         call PetscGetTime(et2,ierr)
         elap2 = elap2 + (et2-st2)
      enddo
      elap2 = elap2/ntest
! Finally do A x I x I
      elap3 = 0.0d0
      do itest = 1,ntest
         call PetscGetTime(st3,ierr)
         do p=1,n
            do q=1,n
               do r=1,n
                  sum = 0.0d0
                  do i=1,n
                     sum = sum + A(p,i,3)*v(r,q,i)
                  enddo
                  w(r,q,p) = w(r,q,p)+sum
               enddo
            enddo
         enddo
         call PetscGetTime(et3,ierr)
         elap3 = elap3 + (et3-st3)
      enddo
      elap3 = elap3/ntest
      flops1 = 2.0d-6*n**(dim+1)/elap1
      flops2 = 2.0d-6*n**(dim+1)/elap2
      flops3 = 2.0d-6*n**(dim+1)/elap3
      flops =  3*2.0d-6*n**(dim+1)/(elap1+elap2+elap3)
      print *,'In subroutine product3D ..........'
      print*,'Flop rate for I x I x A is ',flops1
      print*,'Flop rate for I x A x I is ',flops2
      print*,'Flop rate for A x I x I is ',flops3
      print*,'Total flop rate is ',flops

      return
      end

#undef __FUNCT__
#define __FUNCT__ "product3DMXM"
      subroutine product3DMXM(A,B,v,w)
      implicit none
      double precision A(n,n,dim),B(n*n*dim),v(n,n,n),w(n,n,n),sum
      double precision w1(n,n,n),w2(n,n,n)
      double precision st1,st2,st3,et1,et2,et3,elap1,elap2,elap3
      double precision flops,flops1,flops2,flops3
      integer i,k,p,q,r,ierr,itest
      

! First do I x I x A      
      elap1 = 0.0d0    
      do itest = 1,ntest
         call PetscGetTime(st1,ierr)
         call mxm(A(1,1,1),n,v,n,w,n*n)
         call PetscGetTime(et1,ierr)
         elap1 = elap1 + (et1-st1)
      enddo
      elap1 = elap1/ntest
! Next do I x A x I
      elap2 = 0.0d0
      do itest = 1,ntest
         call PetscGetTime(st2,ierr)
         do p=1,n
            call mxm(v(1,1,p),n,B((n*n+1)),n,w1(1,1,p),n)
         enddo
         call PetscGetTime(et2,ierr)
         elap2 = elap2 + (et2-st2)
      enddo
      elap2 = elap2/ntest
! Finally do A x I x I
      elap3 = 0.0d0
      do itest = 1,ntest
         call PetscGetTime(st3,ierr)
         call mxm(v,n*n,B((2*n*n+1)),n,w2,n)
         call PetscGetTime(et3,ierr)
         elap3 = elap3 + (et3-st3)
      enddo
      elap3 = elap3/ntest
      do p=1,n
        do q=1,n
          do r=1,n
             w(r,q,p) = w(r,q,p)+w1(r,q,p)+w2(r,q,p)
          enddo
       enddo
      enddo
      flops1 = 2.0d-6*n**(dim+1)/elap1
      flops2 = 2.0d-6*n**(dim+1)/elap2
      flops3 = 2.0d-6*n**(dim+1)/elap3
      flops =  3*2.0d-6*n**(dim+1)/(elap1+elap2+elap3)
      print *,'In subroutine product3DMXM .......'
      print*,'Flop rate for I x I x A is ',flops1
      print*,'Flop rate for I x A x I is ',flops2
      print*,'Flop rate for A x I x I is ',flops3
      print*,'Total flop rate is ',flops
!
      return
      end

#undef __FUNCT__
#define __FUNCT__ "product3DNEW"
      subroutine product3DNEW(A,B,v,w)
      implicit none
      double precision A(n,n,dim),B(n*n*dim),v(n,n,n),w(n,n,n),sum
      double precision w1(n,n,n),w2(n,n,n)
      double precision st1,st2,st3,et1,et2,et3,elap1,elap2,elap3
      double precision flops,flops1,flops2,flops3
      integer i,k,p,q,r,ierr,itest
      

! First do I x I x A      
      elap1 = 0.0d0    
      do itest = 1,ntest
         call PetscGetTime(st1,ierr)
         call productIA(n,n*n,A(1,1,1),v,w)
         call PetscGetTime(et1,ierr)
         elap1 = elap1 + (et1-st1)
      enddo
      elap1 = elap1/ntest
! Next do I x A x I
      elap2 = 0.0d0
      do itest = 1,ntest
         call PetscGetTime(st2,ierr)
         call productIAI(n,n,n,B((n*n+1)),v,w1)
         call PetscGetTime(et2,ierr)
         elap2 = elap2 + (et2-st2)
      enddo
      elap2 = elap2/ntest
! Finally do A x I x I
      elap3 = 0.0d0
      do itest = 1,ntest
         call PetscGetTime(st3,ierr)
         call productAI(n,n*n,B((2*n*n+1)),v,w2)
         call PetscGetTime(et3,ierr)
         elap3 = elap3 + (et3-st3)
      enddo
      elap3 = elap3/ntest
      do p=1,n
        do q=1,n
          do r=1,n
             w(r,q,p) = w(r,q,p)+w1(r,q,p)+w2(r,q,p)
          enddo
       enddo
      enddo
      flops1 = 2.0d-6*n**(dim+1)/elap1
      flops2 = 2.0d-6*n**(dim+1)/elap2
      flops3 = 2.0d-6*n**(dim+1)/elap3
      flops =  3*2.0d-6*n**(dim+1)/(elap1+elap2+elap3)
      print *,'In subroutine product3DMXM .......'
      print*,'Flop rate for I x I x A is ',flops1
      print*,'Flop rate for I x A x I is ',flops2
      print*,'Flop rate for A x I x I is ',flops3
      print*,'Total flop rate is ',flops
!
      return
      end

