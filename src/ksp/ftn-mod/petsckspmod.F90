module petsckspdef
  use, intrinsic :: ISO_C_binding
  use petscdmdef

#include <../ftn/ksp/petscall.h>
end module petsckspdef

module petscksp
  use petscdm
  use petsckspdef

#include <../src/ksp/ftn-mod/petscksp.h90>
#include <../ftn/ksp/petscall.h90>

contains

#include <../ftn/ksp/petscall.hf90>

end module petscksp

! Return the address of the PETSC_NULL_KSP_POINTER descriptor, so C stubs can recognize an omitted array argument.
function PETSC_NULL_KSP_POINTER_Fortran() bind(C, name="PETSC_NULL_KSP_POINTER_Fortran") result(ptr)
  use, intrinsic :: ISO_C_binding
  use petsckspdef, only: tKSP, PETSC_NULL_KSP_POINTER
  implicit none
  type(c_ptr) ptr

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_KSP_POINTER_Fortran
#endif
  interface
    subroutine F90Array1dGetDescriptor(array, address)
      use, intrinsic :: ISO_C_binding
      import tKSP
      KSP, pointer :: array(:)
      type(c_ptr) address
    end subroutine
  end interface

  call F90Array1dGetDescriptor(PETSC_NULL_KSP_POINTER, ptr)
end function
