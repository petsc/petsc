      program main
 
      Interface 
         Subroutine foo2(iarray,darray)
         integer, pointer          :: iarray(:,:,:)
         double precision, pointer :: darray(:,:,:)
         End Subroutine
      End Interface
 
 
      integer,pointer :: i(:,:,:)
      integer,target  :: ii(4:13,5:104,13)

      double precision,pointer :: d(:,:,:)
      double precision,target  :: dd(4:13,5:104,13)
 
      i => ii
      d => dd
      i(1,1,1)=10

      call foo1(i,d)
      call foo2(i,d)

      i(1,1,1)=100
      end
