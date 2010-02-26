!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine F90Array1dCreateScalar(array,start,len,ptr)
      implicit none
#include "finclude/petscsys.h"
      PetscInt start,len
      PetscScalar, target :: array(start+len-1)
      PetscScalar, pointer :: ptr(:)

      ptr => array(start:start+len-1)
      end subroutine

      subroutine F90Array1dCreateReal(array,start,len,ptr)
      implicit none
#include "finclude/petscsys.h"
      PetscInt start,len
      PetscReal, target :: array(start+len-1)
      PetscReal, pointer :: ptr(:)

      ptr => array(start:start+len-1)
      end subroutine

      subroutine F90Array1dCreateInt(array,start,len,ptr)
      implicit none
#include "finclude/petscsys.h"
      PetscInt start,len
      PetscInt, target :: array(start+len-1)
      PetscInt, pointer :: ptr(:)

      ptr => array(start:start+len-1)
      end subroutine

      subroutine F90Array1dCreateFortranAddr(array,start,len,ptr)
      implicit none
#include "finclude/petscsys.h"
      PetscInt start,len
      PetscFortranAddr, target :: array(start+len-1)
      PetscFortranAddr, pointer :: ptr(:)

      ptr => array(start:start+len-1)
      end subroutine

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine F90Array1dAccessScalar(ptr,address)
      implicit none
#include "finclude/petscsys.h"
      PetscScalar, pointer :: ptr(:)
      PetscFortranAddr address
      PetscInt start

      start = lbound(ptr,1)
      call F90Array1dGetAddrScalar(ptr(start),address)
      end subroutine

      subroutine F90Array1dAccessReal(ptr,address)
      implicit none
#include "finclude/petscsys.h"
      PetscReal, pointer :: ptr(:)
      PetscFortranAddr address
      PetscInt start

      start = lbound(ptr,1)
      call F90Array1dGetAddrReal(ptr(start),address)
      end subroutine

      subroutine F90Array1dAccessInt(ptr,address)
      implicit none
#include "finclude/petscsys.h"
      PetscInt, pointer :: ptr(:)
      PetscFortranAddr address
      PetscInt start

      start = lbound(ptr,1)
      call F90Array1dGetAddrInt(ptr(start),address)
      end subroutine

      subroutine F90Array1dAccessFortranAddr(ptr,address)
      implicit none
#include "finclude/petscsys.h"
      PetscFortranAddr, pointer :: ptr(:)
      PetscFortranAddr address
      PetscInt start

      start = lbound(ptr,1)
      call F90Array1dGetAddrFortranAddr(ptr(start),address)
      end subroutine
      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine F90Array1dDestroyScalar(ptr)
      implicit none
#include "finclude/petscsys.h"
      PetscScalar, pointer :: ptr(:)

      nullify(ptr)
      end subroutine

      subroutine F90Array1dDestroyReal(ptr)
      implicit none
#include "finclude/petscsys.h"
      PetscReal, pointer :: ptr(:)

      nullify(ptr)
      end subroutine

      subroutine F90Array1dDestroyInt(ptr)
      implicit none
#include "finclude/petscsys.h"
      PetscInt, pointer :: ptr(:)

      nullify(ptr)
      end subroutine

      subroutine F90Array1dDestroyFortranAddr(ptr)
      implicit none
#include "finclude/petscsys.h"
      PetscFortranAddr, pointer :: ptr(:)

      nullify(ptr)
      end subroutine
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine F90Array2dCreateScalar(array,start1,len1,                        &
     &     start2,len2,ptr)
      implicit none
#include "finclude/petscsys.h"
      PetscInt start1,len1
      PetscInt start2,len2
      PetscScalar, target ::                                                      &
     &     array(start1+len1-1,start2+len2-1)
      PetscScalar, pointer :: ptr(:,:)

      ptr => array(start1:len1,start2:len2)
      end subroutine

      subroutine F90Array2dCreateReal(array,start1,len1,                          &
     &     start2,len2,ptr)
      implicit none
#include "finclude/petscsys.h"
      PetscInt start1,len1
      PetscInt start2,len2
      PetscReal, target ::                                                        &
     &     array(start1+len1-1,start2+len2-1)
      PetscReal, pointer :: ptr(:,:)

      ptr => array(start1:len1,start2:len2)
      end subroutine

      subroutine F90Array2dCreateInt(array,start1,len1,                           &
     &     start2,len2,ptr)
      implicit none
#include "finclude/petscsys.h"
      PetscInt start1,len1
      PetscInt start2,len2
      PetscInt, target ::                                                         &
     &     array(start1+len1-1,start2+len2-1)
      PetscInt, pointer :: ptr(:,:)

      ptr => array(start1:len1,start2:len2)
      end subroutine

      subroutine F90Array2dCreateFortranAddr(array,start1,len1,                   &
     &     start2,len2,ptr)
      implicit none
#include "finclude/petscsys.h"
      PetscInt start1,len1
      PetscInt start2,len2
      PetscFortranAddr, target ::                                                 &
     &     array(start1+len1-1,start2+len2-1)
      PetscFortranAddr, pointer :: ptr(:,:)

      ptr => array(start1:len1,start2:len2)
      end subroutine

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine F90Array2dAccessScalar(ptr,address)
      implicit none
#include "finclude/petscsys.h"
      PetscScalar, pointer :: ptr(:,:)
      PetscFortranAddr address
      PetscInt start1,start2

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      call F90Array2dGetAddrScalar(ptr(start1,start2),address)
      end subroutine

      subroutine F90Array2dAccessReal(ptr,address)
      implicit none
#include "finclude/petscsys.h"
      PetscReal, pointer :: ptr(:,:)
      PetscFortranAddr address
      PetscInt start1,start2

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      call F90Array2dGetAddrReal(ptr(start1,start2),address)
      end subroutine

      subroutine F90Array2dAccessInt(ptr,address)
      implicit none
#include "finclude/petscsys.h"
      PetscInt, pointer :: ptr(:,:)
      PetscFortranAddr address
      PetscInt start1,start2

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      call F90Array2dGetAddrInt(ptr(start1,start2),address)
      end subroutine

      subroutine F90Array2dAccessFortranAddr(ptr,address)
      implicit none
#include "finclude/petscsys.h"
      PetscFortranAddr, pointer :: ptr(:,:)
      PetscFortranAddr address
      PetscInt start1,start2

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      call F90Array2dGetAddrFortranAddr(ptr(start1,start2),address)
      end subroutine
      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine F90Array2dDestroyScalar(ptr)
      implicit none
#include "finclude/petscsys.h"
      PetscScalar, pointer :: ptr(:,:)

      nullify(ptr)
      end subroutine

      subroutine F90Array2dDestroyReal(ptr)
      implicit none
#include "finclude/petscsys.h"
      PetscReal, pointer :: ptr(:,:)

      nullify(ptr)
      end subroutine

      subroutine F90Array2dDestroyInt(ptr)
      implicit none
#include "finclude/petscsys.h"
      PetscInt, pointer :: ptr(:,:)

      nullify(ptr)
      end subroutine

      subroutine F90Array2dDestroyFortranAddr(ptr)
      implicit none
#include "finclude/petscsys.h"
      PetscFortranAddr, pointer :: ptr(:,:)

      nullify(ptr)
      end subroutine
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
