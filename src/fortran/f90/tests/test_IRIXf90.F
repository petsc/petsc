
      program main
 
      Interface 
         Subroutine foo2(iarray,darray)
         integer, pointer          :: iarray(:,:,:,:)
         double precision, pointer :: darray(:,:)
         End Subroutine
      End Interface
 
 
      integer,pointer :: i(:,:,:,:)
      integer,target  :: ii(2:11,13:112,4:13,1000)

      double precision,pointer :: d(:,:)
      double precision,target  :: dd(4:13,5)
 
      i => ii
      d => dd
      i(2,13,4,1)=10

      call foo1(i,d)
      call foo2(i,d)
      i(2,13,4,1)=100

      end
