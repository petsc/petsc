#undef __FUNCT__
#define __FUNCT__ "productIA"
      subroutine productIA(n1,n2,A,v,w)
      implicit none
      integer n1,n2
      double precision A(n1,n1),v(n1,n2),w(n1,n2)
      call mxm(A,n1,v,n1,w,n2)
      return
      end

#undef __FUNCT__
#define __FUNCT__ "productAI"
      subroutine productAI(n1,n2,A,v,w)
      implicit none
      integer n1,n2
      double precision A(n1,n1),v(n2,n1),w(n2,n1)
      call mxm(v,n2,A,n1,w,n1)
      return
      end

#undef __FUNCT__
#define __FUNCT__ "productIAI"
      subroutine productIAI(n1,n2,n3,A,v,w)
      implicit none
      integer n1,n2,n3,i
      double precision A(n2,n2),v(n3,n2,n1),w(n2,n3,n1)
      call mxm(v,n2,A,n1,w,n1)
      do i=1,n1
         call mxm(v(1,1,i),n3,A,n2,w(1,1,i),n2)
      enddo
      return
      end
