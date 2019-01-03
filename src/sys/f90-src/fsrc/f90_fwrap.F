!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#include <petsc/finclude/petscsys.h>
      subroutine F90Array1dCreateScalar(array,start,len1,ptr)
      implicit none
      PetscInt start,len1
      PetscScalar, target ::                                                      &
     &     array(start:start+len1-1)
      PetscScalar, pointer :: ptr(:)

      ptr => array
      end subroutine

      subroutine F90Array1dCreateReal(array,start,len1,ptr)
      implicit none
      PetscInt start,len1
      PetscReal, target ::                                                        &
     &     array(start:start+len1-1)
      PetscReal, pointer :: ptr(:)

      ptr => array
      end subroutine

      subroutine F90Array1dCreateInt(array,start,len1,ptr)
      implicit none
      PetscInt start,len1
      PetscInt, target ::                                                         &
     &     array(start:start+len1-1)
      PetscInt, pointer :: ptr(:)

      ptr => array
      end subroutine

      subroutine F90Array1dCreateFortranAddr(array,start,len1,ptr)
      implicit none
      PetscInt start,len1
      PetscFortranAddr, target ::                                                 &
     &     array(start:start+len1-1)
      PetscFortranAddr, pointer :: ptr(:)

      ptr => array
      end subroutine

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine F90Array1dAccessScalar(ptr,address)
      implicit none
      PetscScalar, pointer :: ptr(:)
      PetscFortranAddr address
      PetscInt start

      if (associated(ptr) .eqv. .false.) then
        address = 0
      else
        start = lbound(ptr,1)
        call F90Array1dGetAddrScalar(ptr(start),address)
      endif
      end subroutine

      subroutine F90Array1dAccessReal(ptr,address)
      implicit none
      PetscReal, pointer :: ptr(:)
      PetscFortranAddr address
      PetscInt start

      if (associated(ptr) .eqv. .false.) then
        address = 0
      else
        start = lbound(ptr,1)
        call F90Array1dGetAddrReal(ptr(start),address)
      endif
      end subroutine

      subroutine F90Array1dAccessInt(ptr,address)
      implicit none
      PetscInt, pointer :: ptr(:)
      PetscFortranAddr address
      PetscInt start

      if (associated(ptr) .eqv. .false.) then
        address = 0
      else
        start = lbound(ptr,1)
        call F90Array1dGetAddrInt(ptr(start),address)
      endif
      end subroutine

      subroutine F90Array1dAccessFortranAddr(ptr,address)
      implicit none
      PetscFortranAddr, pointer :: ptr(:)
      PetscFortranAddr address
      PetscInt start

      if (associated(ptr) .eqv. .false.) then
        address = 0
      else
        start = lbound(ptr,1)
        call F90Array1dGetAddrFortranAddr(ptr(start),address)
      endif
      end subroutine

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine F90Array1dDestroyScalar(ptr)
      implicit none
      PetscScalar, pointer :: ptr(:)

      nullify(ptr)
      end subroutine

      subroutine F90Array1dDestroyReal(ptr)
      implicit none
      PetscReal, pointer :: ptr(:)

      nullify(ptr)
      end subroutine

      subroutine F90Array1dDestroyInt(ptr)
      implicit none
      PetscInt, pointer :: ptr(:)

      nullify(ptr)
      end subroutine

      subroutine F90Array1dDestroyFortranAddr(ptr)
      implicit none
      PetscFortranAddr, pointer :: ptr(:)

      nullify(ptr)
      end subroutine
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine F90Array2dCreateScalar(array,start1,len1,                        &
     &     start2,len2,ptr)
      implicit none
      PetscInt start1,len1
      PetscInt start2,len2
      PetscScalar, target ::                                                      &
     &     array(start1:start1+len1-1,start2:start2+len2-1)
      PetscScalar, pointer :: ptr(:,:)

      ptr => array
      end subroutine

      subroutine F90Array2dCreateReal(array,start1,len1,                          &
     &     start2,len2,ptr)
      implicit none
      PetscInt start1,len1
      PetscInt start2,len2
      PetscReal, target ::                                                        &
     &     array(start1:start1+len1-1,start2:start2+len2-1)
      PetscReal, pointer :: ptr(:,:)

      ptr => array
      end subroutine

      subroutine F90Array2dCreateInt(array,start1,len1,                           &
     &     start2,len2,ptr)
      implicit none
      PetscInt start1,len1
      PetscInt start2,len2
      PetscInt, target ::                                                         &
     &     array(start1:start1+len1-1,start2:start2+len2-1)
      PetscInt, pointer :: ptr(:,:)

      ptr => array
      end subroutine

      subroutine F90Array2dCreateFortranAddr(array,start1,len1,                   &
     &     start2,len2,ptr)
      implicit none
      PetscInt start1,len1
      PetscInt start2,len2
      PetscFortranAddr, target ::                                                 &
     &     array(start1:start1+len1-1,start2:start2+len2-1)
      PetscFortranAddr, pointer :: ptr(:,:)

      ptr => array
      end subroutine

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine F90Array2dAccessScalar(ptr,address)
      implicit none
      PetscScalar, pointer :: ptr(:,:)
      PetscFortranAddr address
      PetscInt start1,start2

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      call F90Array2dGetAddrScalar(ptr(start1,start2),address)
      end subroutine

      subroutine F90Array2dAccessReal(ptr,address)
      implicit none
      PetscReal, pointer :: ptr(:,:)
      PetscFortranAddr address
      PetscInt start1,start2

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      call F90Array2dGetAddrReal(ptr(start1,start2),address)
      end subroutine

      subroutine F90Array2dAccessInt(ptr,address)
      implicit none
      PetscInt, pointer :: ptr(:,:)
      PetscFortranAddr address
      PetscInt start1,start2

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      call F90Array2dGetAddrInt(ptr(start1,start2),address)
      end subroutine

      subroutine F90Array2dAccessFortranAddr(ptr,address)
      implicit none
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
      PetscScalar, pointer :: ptr(:,:)

      nullify(ptr)
      end subroutine

      subroutine F90Array2dDestroyReal(ptr)
      implicit none
      PetscReal, pointer :: ptr(:,:)

      nullify(ptr)
      end subroutine

      subroutine F90Array2dDestroyInt(ptr)
      implicit none
      PetscInt, pointer :: ptr(:,:)

      nullify(ptr)
      end subroutine

      subroutine F90Array2dDestroyFortranAddr(ptr)
      implicit none
      PetscFortranAddr, pointer :: ptr(:,:)

      nullify(ptr)
      end subroutine
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine F90Array3dCreateScalar(array,start1,len1,                        &
     &     start2,len2,start3,len3,ptr)
      implicit none
      PetscInt start1,len1
      PetscInt start2,len2
      PetscInt start3,len3
      PetscScalar, target ::                                                      &
     &     array(start1:start1+len1-1,start2:start2+len2-1,                       &
     &           start3:start3+len3-1)
      PetscScalar, pointer :: ptr(:,:,:)

      ptr => array
      end subroutine

      subroutine F90Array3dCreateReal(array,start1,len1,                          &
     &     start2,len2,start3,len3,ptr)
      implicit none
      PetscInt start1,len1
      PetscInt start2,len2
      PetscInt start3,len3
      PetscReal, target ::                                                        &
     &     array(start1:start1+len1-1,start2:start2+len2-1,                       &
     &           start3:start3+len3-1)
      PetscReal, pointer :: ptr(:,:,:)

      ptr => array
      end subroutine

      subroutine F90Array3dCreateInt(array,start1,len1,                           &
     &     start2,len2,start3,len3,ptr)
      implicit none
      PetscInt start1,len1
      PetscInt start2,len2
      PetscInt start3,len3
      PetscInt, target ::                                                         &
     &     array(start1:start1+len1-1,start2:start2+len2-1,                       &
     &           start3:start3+len3-1)
      PetscInt, pointer :: ptr(:,:,:)

      ptr => array
      end subroutine

      subroutine F90Array3dCreateFortranAddr(array,start1,len1,                   &
     &     start2,len2,start3,len3,ptr)
      implicit none
      PetscInt start1,len1
      PetscInt start2,len2
      PetscInt start3,len3
      PetscFortranAddr, target ::                                                 &
     &     array(start1:start1+len1-1,start2:start2+len2-1,                       &
     &           start3:start3+len3-1)
      PetscFortranAddr, pointer :: ptr(:,:,:)

      ptr => array
      end subroutine

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine F90Array3dAccessScalar(ptr,address)
      implicit none
      PetscScalar, pointer :: ptr(:,:,:)
      PetscFortranAddr address
      PetscInt start1,start2,start3

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      start3 = lbound(ptr,3)
      call F90Array3dGetAddrScalar(ptr(start1,start2,start3),address)
      end subroutine

      subroutine F90Array3dAccessReal(ptr,address)
      implicit none
      PetscReal, pointer :: ptr(:,:,:)
      PetscFortranAddr address
      PetscInt start1,start2,start3

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      start3 = lbound(ptr,3)
      call F90Array3dGetAddrReal(ptr(start1,start2,start3),address)
      end subroutine

      subroutine F90Array3dAccessInt(ptr,address)
      implicit none
      PetscInt, pointer :: ptr(:,:,:)
      PetscFortranAddr address
      PetscInt start1,start2,start3

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      start3 = lbound(ptr,3)
      call F90Array3dGetAddrInt(ptr(start1,start2,start3),address)
      end subroutine

      subroutine F90Array3dAccessFortranAddr(ptr,address)
      implicit none
      PetscFortranAddr, pointer :: ptr(:,:,:)
      PetscFortranAddr address
      PetscInt start1,start2,start3

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      start3 = lbound(ptr,3)
      call F90Array3dGetAddrFortranAddr(ptr(start1,start2,start3),        &
     &                                  address)
      end subroutine

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine F90Array3dDestroyScalar(ptr)
      implicit none
      PetscScalar, pointer :: ptr(:,:,:)

      nullify(ptr)
      end subroutine

      subroutine F90Array3dDestroyReal(ptr)
      implicit none
      PetscReal, pointer :: ptr(:,:,:)

      nullify(ptr)
      end subroutine

      subroutine F90Array3dDestroyInt(ptr)
      implicit none
      PetscInt, pointer :: ptr(:,:,:)

      nullify(ptr)
      end subroutine

      subroutine F90Array3dDestroyFortranAddr(ptr)
      implicit none
      PetscFortranAddr, pointer :: ptr(:,:,:)

      nullify(ptr)
      end subroutine

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine F90Array4dCreateScalar(array,start1,len1,                        &
     &     start2,len2,start3,len3,start4,len4,ptr)
      implicit none
      PetscInt start1,len1
      PetscInt start2,len2
      PetscInt start3,len3
      PetscInt start4,len4
      PetscScalar, target ::                                                      &
     &     array(start1:start1+len1-1,start2:start2+len2-1,                       &
     &           start3:start3+len3-1,start4:start4+len4-1)
      PetscScalar, pointer :: ptr(:,:,:,:)

      ptr => array
      end subroutine

      subroutine F90Array4dCreateReal(array,start1,len1,                          &
     &     start2,len2,start3,len3,start4,len4,ptr)
      implicit none
      PetscInt start1,len1
      PetscInt start2,len2
      PetscInt start3,len3
      PetscInt start4,len4
      PetscReal, target ::                                                        &
     &     array(start1:start1+len1-1,start2:start2+len2-1,                       &
     &           start3:start3+len3-1,start4:start4+len4-1)
      PetscReal, pointer :: ptr(:,:,:,:)

      ptr => array
      end subroutine

      subroutine F90Array4dCreateInt(array,start1,len1,                           &
     &     start2,len2,start3,len3,start4,len4,ptr)
      implicit none
      PetscInt start1,len1
      PetscInt start2,len2
      PetscInt start3,len3
      PetscInt start4,len4
      PetscInt, target ::                                                         &
     &     array(start1:start1+len1-1,start2:start2+len2-1,                       &
     &           start3:start3+len3-1,start4:start4+len4-1)
      PetscInt, pointer :: ptr(:,:,:,:)

      ptr => array
      end subroutine

      subroutine F90Array4dCreateFortranAddr(array,start1,len1,                   &
     &     start2,len2,start3,len3,start4,len4,ptr)
      implicit none
      PetscInt start1,len1
      PetscInt start2,len2
      PetscInt start3,len3
      PetscInt start4,len4
      PetscFortranAddr, target ::                                                 &
     &     array(start1:start1+len1-1,start2:start2+len2-1,                       &
     &           start3:start3+len3-1,start4:start4+len4-1)
      PetscFortranAddr, pointer :: ptr(:,:,:,:)

      ptr => array
      end subroutine

      subroutine F90Array4dAccessScalar(ptr,address)
      implicit none
      PetscScalar, pointer :: ptr(:,:,:,:)
      PetscFortranAddr address
      PetscInt start1,start2,start3,start4

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      start3 = lbound(ptr,3)
      start4 = lbound(ptr,4)
      call F90Array4dGetAddrScalar(ptr(start1,start2,start3,start4),              &
     &                             address)
      end subroutine

      subroutine F90Array4dAccessReal(ptr,address)
      implicit none
      PetscReal, pointer :: ptr(:,:,:,:)
      PetscFortranAddr address
      PetscInt start1,start2,start3,start4

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      start3 = lbound(ptr,3)
      start4 = lbound(ptr,4)
      call F90Array4dGetAddrReal(ptr(start1,start2,start3,start4),                &
     &                             address)
      end subroutine

      subroutine F90Array4dAccessInt(ptr,address)
      implicit none
      PetscInt, pointer :: ptr(:,:,:,:)
      PetscFortranAddr address
      PetscInt start1,start2,start3,start4

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      start3 = lbound(ptr,3)
      start4 = lbound(ptr,4)
      call F90Array4dGetAddrInt(ptr(start1,start2,start3,start4),                 &
     &                             address)
      end subroutine

      subroutine F90Array4dAccessFortranAddr(ptr,address)
      implicit none
      PetscScalar, pointer :: ptr(:,:,:,:)
      PetscFortranAddr address
      PetscFortranAddr start1,start2,start3,start4

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      start3 = lbound(ptr,3)
      start4 = lbound(ptr,4)
      call F90Array4dGetAddrFortranAddr(ptr(start1,start2,start3,                 &
     &                                      start4),address)
      end subroutine

      subroutine F90Array4dDestroyScalar(ptr)
      implicit none
      PetscScalar, pointer :: ptr(:,:,:,:)

      nullify(ptr)
      end subroutine

      subroutine F90Array4dDestroyReal(ptr)
      implicit none
      PetscReal, pointer :: ptr(:,:,:,:)

      nullify(ptr)
      end subroutine

      subroutine F90Array4dDestroyInt(ptr)
      implicit none
      PetscInt, pointer :: ptr(:,:,:,:)

      nullify(ptr)
      end subroutine

      subroutine F90Array4dDestroyFortranAddr(ptr)
      implicit none
      PetscFortranAddr, pointer :: ptr(:,:,:,:)

      nullify(ptr)
      end subroutine

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

