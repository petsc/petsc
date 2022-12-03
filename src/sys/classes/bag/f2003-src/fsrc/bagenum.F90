
#include "petsc/finclude/petscsys.h"

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PetscBagRegisterEnum
#endif
      Subroutine PetscBagRegisterEnum(bag,addr,FArray,def,n,h,ierr)
      use,intrinsic :: iso_c_binding
      use petscsys
      implicit none

      PetscBag   bag
      character(*)                n,h
      character(*)                FArray(*)
      PetscEnum                   :: def
      PetscErrorCode,intent(out)  :: ierr
      PetscReal addr(*)

      Type(C_Ptr),Dimension(:),Pointer :: CArray
      character(kind=c_char),pointer   :: nullc => null()
      PetscInt   :: i,Len
      Character(kind=C_char,len=256),Dimension(:),Pointer::list1

      do i=1,256
        if (len_trim(Farray(i)) .eq. 0) then
          Len = i-1
          goto 100
        endif
        if (len_trim(Farray(i)) .gt. 255) then
          ierr = PETSC_ERR_ARG_OUTOFRANGE
          return
        endif
      enddo
      ierr = PETSC_ERR_ARG_OUTOFRANGE
      return

 100  continue

      Allocate(list1(Len),stat=ierr)
      if (ierr .ne. 0) return
      Allocate(CArray(Len+1),stat=ierr)
      if (ierr .ne. 0) return

      do i=1,Len
         list1(i) = trim(FArray(i))//C_NULL_CHAR
         CArray(i) = c_loc(list1(i))
      enddo

      CArray(Len+1) = c_loc(nullc)
      call PetscBagRegisterEnumPrivate(bag,addr,CArray,def,n,h,ierr)
      DeAllocate(CArray)
      DeAllocate(list1)
      End Subroutine
