module petscmatdef
  use, intrinsic :: ISO_C_binding
  use petscvecdef
#include "petsc/finclude/petscmat.h"
#include "petsc/finclude/petscmatcoarsen.h"
#include "petsc/finclude/petscpartitioner.h"
#include "petsc/finclude/petscmathypre.h"
#include "petsc/finclude/petscmathtool.h"
#include "petsc/finclude/petscmatelemental.h"
#include <../ftn/mat/petscmat.h>
#include <../ftn/mat/petscmatcoarsen.h>
#include <../ftn/mat/petscpartitioner.h>

end module

module petscmat
  use petscmatdef
  use petscvec

#include <../src/mat/ftn-mod/petscmat.h90>
#include <../ftn/mat/petscmat.h90>
#include <../ftn/mat/petscmatcoarsen.h90>
#include <../ftn/mat/petscpartitioner.h90>

!     deprecated functions

  interface MatDenseGetArrayF90
    module procedure MatDenseGetArrayF901d, MatDenseGetArrayF902d
  end interface

  interface MatDenseRestoreArrayF90
    module procedure MatDenseRestoreArrayF901d, MatDenseRestoreArrayF902d
  end interface

  interface MatDenseGetArrayReadF90
    module procedure MatDenseGetArrayReadF901d, MatDenseGetArrayReadF902d
  end interface

  interface MatDenseRestoreArrayReadF90
    module procedure MatDenseRestoreArrayWriteF901d, MatDenseRestoreArrayWriteF902d
  end interface

  interface MatDenseGetArrayWriteF90
    module procedure MatDenseGetArrayWriteF901d, MatDenseGetArrayWriteF902d
  end interface

  interface MatDenseRestoreArrayWriteF90
    module procedure MatDenseRestoreArrayWriteF901d, MatDenseRestoreArrayWriteF902d
  end interface

contains

#include <../ftn/mat/petscmat.hf90>
#include <../ftn/mat/petscmatcoarsen.hf90>
#include <../ftn/mat/petscpartitioner.hf90>

  subroutine MatDenseGetArrayF901d(A, array, ierr)
#if defined(PETSC_DLL_EXPORTS)
!DEC$ ATTRIBUTES DLLEXPORT::MatDenseGetArrayF901d
#endif
    PetscScalar, pointer :: array(:)
    PetscErrorCode ierr
    Mat A
    call MatDenseGetArray(A, array, ierr)
  end subroutine

  subroutine MatDenseRestoreArrayF901d(A, array, ierr)
#if defined(PETSC_DLL_EXPORTS)
!DEC$ ATTRIBUTES DLLEXPORT::MatDenseRestoreArrayF901d
#endif
    PetscScalar, pointer :: array(:)
    PetscErrorCode ierr
    Mat A
    call MatDenseRestoreArray(A, array, ierr)
  end subroutine

  subroutine MatDenseGetArrayReadF901d(A, array, ierr)
#if defined(PETSC_DLL_EXPORTS)
!DEC$ ATTRIBUTES DLLEXPORT::MatDenseGetArrayReadF901d
#endif
    PetscScalar, pointer :: array(:)
    PetscErrorCode ierr
    Mat A
    call MatDenseGetArrayRead(A, array, ierr)
  end subroutine

  subroutine MatDenseRestoreArrayReadF901d(A, array, ierr)
#if defined(PETSC_DLL_EXPORTS)
!DEC$ ATTRIBUTES DLLEXPORT::MatDenseRestoreArrayReadF901d
#endif
    PetscScalar, pointer :: array(:)
    PetscErrorCode ierr
    Mat A
    call MatDenseRestoreArrayRead(A, array, ierr)
  end subroutine

  subroutine MatDenseGetArrayWriteF901d(A, array, ierr)
#if defined(PETSC_DLL_EXPORTS)
!DEC$ ATTRIBUTES DLLEXPORT::MatDenseGetArrayWriteF901d
#endif
    PetscScalar, pointer :: array(:)
    PetscErrorCode ierr
    Mat A
    call MatDenseGetArrayWrite(A, array, ierr)
  end subroutine

  subroutine MatDenseRestoreArrayWriteF901d(A, array, ierr)
#if defined(PETSC_DLL_EXPORTS)
!DEC$ ATTRIBUTES DLLEXPORT::MatDenseRestoreArrayWriteF901d
#endif
    PetscScalar, pointer :: array(:)
    PetscErrorCode ierr
    Mat A
    call MatDenseRestoreArrayWrite(A, array, ierr)
  end subroutine

  subroutine MatDenseGetArrayF902d(A, array, ierr)
#if defined(PETSC_DLL_EXPORTS)
!DEC$ ATTRIBUTES DLLEXPORT::MatDenseGetArrayF902d
#endif
    PetscScalar, pointer :: array(:, :)
    PetscErrorCode ierr
    Mat A
    call MatDenseGetArray(A, array, ierr)
  end subroutine

  subroutine MatDenseRestoreArrayF902d(A, array, ierr)
#if defined(PETSC_DLL_EXPORTS)
!DEC$ ATTRIBUTES DLLEXPORT::MatDenseRestoreArrayF902d
#endif
    PetscScalar, pointer :: array(:, :)
    PetscErrorCode ierr
    Mat A
    call MatDenseRestoreArray(A, array, ierr)
  end subroutine

  subroutine MatDenseGetArrayReadF902d(A, array, ierr)
#if defined(PETSC_DLL_EXPORTS)
!DEC$ ATTRIBUTES DLLEXPORT::MatDenseGetArrayReadF902d
#endif
    PetscScalar, pointer :: array(:, :)
    PetscErrorCode ierr
    Mat A
    call MatDenseGetArrayRead(A, array, ierr)
  end subroutine

  subroutine MatDenseRestoreArrayReadF902d(A, array, ierr)
#if defined(PETSC_DLL_EXPORTS)
!DEC$ ATTRIBUTES DLLEXPORT::MatDenseRestoreArrayReadF902d
#endif
    PetscScalar, pointer :: array(:, :)
    PetscErrorCode ierr
    Mat A
    call MatDenseRestoreArrayRead(A, array, ierr)
  end subroutine

  subroutine MatDenseGetArrayWriteF902d(A, array, ierr)
#if defined(PETSC_DLL_EXPORTS)
!DEC$ ATTRIBUTES DLLEXPORT::MatDenseGetArrayWriteF902d
#endif
    PetscScalar, pointer :: array(:, :)
    PetscErrorCode ierr
    Mat A
    call MatDenseGetArrayWrite(A, array, ierr)
  end subroutine

  subroutine MatDenseRestoreArrayWriteF902d(A, array, ierr)
#if defined(PETSC_DLL_EXPORTS)
!DEC$ ATTRIBUTES DLLEXPORT::MatDenseRestoreArrayWriteF902d
#endif
    PetscScalar, pointer :: array(:, :)
    PetscErrorCode ierr
    Mat A
    call MatDenseRestoreArrayWrite(A, array, ierr)
  end subroutine

end module

! Return the address of the PETSC_NULL_MAT_POINTER descriptor, so C stubs can recognize an omitted array argument.
function PETSC_NULL_MAT_POINTER_Fortran() bind(C, name="PETSC_NULL_MAT_POINTER_Fortran") result(ptr)
#if defined(PETSC_DLL_EXPORTS)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_MAT_POINTER_Fortran
#endif
  use, intrinsic :: ISO_C_binding
  use petscmatdef, only: tMat, PETSC_NULL_MAT_POINTER
  implicit none
  type(c_ptr) ptr

  interface
    subroutine F90Array1dGetDescriptor(array, address)
      use, intrinsic :: ISO_C_binding
      import tMat
      Mat, pointer :: array(:)
      type(c_ptr) address
    end subroutine
  end interface

  call F90Array1dGetDescriptor(PETSC_NULL_MAT_POINTER, ptr)
end function
