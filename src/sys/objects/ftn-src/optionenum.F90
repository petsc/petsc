#include "petsc/finclude/petscsys.h"

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PetscOptionsGetEnum
!DEC$ ATTRIBUTES DLLEXPORT::PetscOptionsEnum
#endif

Subroutine PetscOptionsGetEnum(po, pre, name, FArray, opt, set, ierr)
  use, intrinsic :: iso_c_binding
  use petscsysdef
  implicit none

  character(*) pre, name
  character(*) FArray(*)
  PetscEnum                   :: opt
  PetscBool                   :: set
  PetscOptions                :: po
  PetscErrorCode, intent(out)  :: ierr

  Type(C_Ptr), Dimension(:), Pointer :: CArray
  character(kind=c_char), pointer   :: nullc => null()
  PetscInt   :: i, Len
  Character(kind=C_char, len=99), Dimension(:), Pointer::list1

  Len = 0
  do i = 1, 100
    if (len_trim(Farray(i)) == 0) then
      Len = i - 1
      goto 100
    end if
  end do
100 continue

  Allocate (list1(Len), stat=ierr)
  if (ierr /= 0) return
  Allocate (CArray(Len + 1), stat=ierr)
  if (ierr /= 0) return
  do i = 1, Len
    list1(i) = trim(FArray(i))//C_NULL_CHAR
    CArray(i) = c_loc(list1(i))
  end do

  CArray(Len + 1) = c_loc(nullc)
  call PetscOptionsGetEnumPrivate(po, pre, name, CArray, opt, set, ierr)
  DeAllocate (CArray)
  DeAllocate (list1)
End Subroutine

Subroutine PetscOptionsEnum(opt, text, man, Flist, curr, ivalue, set, ierr)
  use, intrinsic :: iso_c_binding
  use petscsysdef
  implicit none

  character(*) opt, text, man
  character(*) Flist(*)
  PetscEnum                   :: curr, ivalue
  PetscBool                   :: set
  PetscErrorCode, intent(out)  :: ierr

  Type(C_Ptr), Dimension(:), Pointer :: CArray
  character(kind=c_char), pointer   :: nullc => null()
  PetscInt   :: i, Len
  Character(kind=C_char, len=99), Dimension(:), Pointer::list1

  Len = 0
  do i = 1, 100
    if (len_trim(Flist(i)) == 0) then
      Len = i - 1
      goto 100
    end if
  end do
100 continue

  Allocate (list1(Len), stat=ierr)
  if (ierr /= 0) return
  Allocate (CArray(Len + 1), stat=ierr)
  if (ierr /= 0) return
  do i = 1, Len
    list1(i) = trim(Flist(i))//C_NULL_CHAR
    CArray(i) = c_loc(list1(i))
  end do

  CArray(Len + 1) = c_loc(nullc)
  call PetscOptionsEnumPrivate(opt, text, man, CArray, curr, ivalue, set, ierr)

  DeAllocate (CArray)
  DeAllocate (list1)
End Subroutine PetscOptionsEnum
