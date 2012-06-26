
#include "finclude/petscdef.h"
      Subroutine PetscOptionsGetEnum(pre,name,FArray,opt,set,ierr)
      use,intrinsic :: iso_c_binding
      implicit none

      character(*)                pre,name
      character(*)                FArray(*)
      PetscEnum                   :: opt
      PetscBool                   :: set
      PetscErrorCode,intent(out)  :: ierr

      Type(C_Ptr),Dimension(:),Pointer :: CArray
      character(kind=c_char),pointer   :: nullc => null()
      PetscInt   :: i,Len
      Character(kind=C_char,len=99),Dimension(:),Pointer::list1

      Len=0
      do i=1,100
        if (len_trim(Farray(i)) .eq. 0) then
          Len = i-1
          goto 100
        endif
      enddo
 100  continue

      Allocate(list1(Len),stat=ierr);
      Allocate(CArray(Len+1),stat=ierr)
      do i=1,Len
         list1(i) = trim(FArray(i))//C_NULL_CHAR
      enddo

      CArray = (/(c_loc(list1(i)),i=1,Len),c_loc(nullc)/)
      call PetscOptionsGetEnumPrivate(pre,name,CArray,opt,set,ierr)
      DeAllocate(CArray)
      DeAllocate(list1)
      End Subroutine 

