module petscisdef
  use, intrinsic :: ISO_C_binding
  use petscsysdef
#include <petsc/finclude/petscis.h>
#include <../ftn/vec/petscis.h>
#include <petsc/finclude/petscsf.h>
#include <../ftn/vec/petscsf.h>
#include <petsc/finclude/petscsection.h>
#include <../ftn/vec/petscsection.h>

end module

!     Needed by Fortran stub petscsfgetgraph_()
subroutine F90Array1dCreateSFNode(array, start, len, ptr)
  use petscisdef
  implicit none
  PetscInt start, len
  PetscSFNode, target :: array(start:start + len - 1)
  PetscSFNode, pointer :: ptr(:)
  ptr => array
end subroutine
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: F90Array1dCreateSFNode
#endif

subroutine F90Array1dDestroySFNode(ptr)
  use petscisdef
  implicit none
  PetscSFNode, pointer :: ptr(:)
  nullify (ptr)
end subroutine
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: F90Array1dDestroySFNode
#endif

module petscis
  use petscisdef
  use petscsys

  interface PetscSFDestroyRemoteOffsets
    subroutine PetscSFDestroyRemoteOffsets(ptr, ierr)
      use petscisdef
      implicit none
      PetscInt, pointer :: ptr(:)
      PetscErrorCode :: ierr
    end subroutine PetscSFDestroyRemoteOffsets
  end interface

#include <../src/vec/ftn-mod/petscis.h90>
#include <../ftn/vec/petscsf.h90>
#include <../ftn/vec/petscsection.h90>
#include <../ftn/vec/petscis.h90>

contains

#include <../ftn/vec/petscsf.hf90>
#include <../ftn/vec/petscsection.hf90>
#include <../ftn/vec/petscis.hf90>

end module

module petscvecdef
  use petscisdef
#include <petsc/finclude/petscvec.h>
#include <../ftn/vec/petscvec.h>
end module

module petscvec
  use petscis
  use petscvecdef

#include <../src/vec/ftn-mod/petscvec.h90>
#include <../ftn/vec/petscvec.h90>

contains

#include <../ftn/vec/petscvec.hf90>

end module

module petscaodef
  use petscsys
  use petscvecdef
#include <petsc/finclude/petscao.h>
#include <../ftn/vec/petscao.h>
end module

module petscao
  use petscsys
  use petscaodef
#include <../ftn/vec/petscao.h90>
contains
#include <../ftn/vec/petscao.hf90>
end module

module petscpfdef
  use petscsys
  use petscvecdef
#include <petsc/finclude/petscpf.h>
#include <../ftn/vec/petscpf.h>
end module

module petscpf
  use petscsys
  use petscpfdef
#include <../ftn/vec/petscpf.h90>
contains
#include <../ftn/vec/petscpf.hf90>
end module

! Return the address of the PETSC_NULL_IS_POINTER descriptor, so C stubs can recognize an omitted array argument.
function PETSC_NULL_IS_POINTER_Fortran() bind(C, name="PETSC_NULL_IS_POINTER_Fortran") result(ptr)
  use, intrinsic :: ISO_C_binding
  use petscisdef, only: tIS, PETSC_NULL_IS_POINTER
  implicit none
  type(c_ptr) ptr

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_IS_POINTER_Fortran
#endif
  interface
    subroutine F90Array1dGetDescriptor(array, address)
      use, intrinsic :: ISO_C_binding
      import tIS
      IS, pointer :: array(:)
      type(c_ptr) address
    end subroutine
  end interface

  call F90Array1dGetDescriptor(PETSC_NULL_IS_POINTER, ptr)
end function
