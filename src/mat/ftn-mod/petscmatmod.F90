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

!     ----------------------------------------------

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

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseGetArrayF901d
#endif
  Subroutine MatDenseGetArrayF901d(v, array, ierr)
    PetscScalar, pointer :: array(:)
    PetscErrorCode ierr
    Mat v
    call MatDenseGetArray(v, array, ierr)
  End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseRestoreArrayF901d
#endif
  Subroutine MatDenseRestoreArrayF901d(v, array, ierr)
    PetscScalar, pointer :: array(:)
    PetscErrorCode ierr
    Mat v
    call MatDenseRestoreArray(v, array, ierr)
  End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseGetArrayReadF901d
#endif
  Subroutine MatDenseGetArrayReadF901d(v, array, ierr)
    PetscScalar, pointer :: array(:)
    PetscErrorCode ierr
    Mat v
    call MatDenseGetArrayRead(v, array, ierr)
  End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseRestoreArrayReadF901d
#endif
  Subroutine MatDenseRestoreArrayReadF901d(v, array, ierr)
    PetscScalar, pointer :: array(:)
    PetscErrorCode ierr
    Mat v
    call MatDenseRestoreArrayRead(v, array, ierr)
  End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseGetArrayWriteF901d
#endif
  Subroutine MatDenseGetArrayWriteF901d(v, array, ierr)
    PetscScalar, pointer :: array(:)
    PetscErrorCode ierr
    Mat v
    call MatDenseGetArrayWrite(v, array, ierr)
  End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseRestoreArrayWriteF901d
#endif
  Subroutine MatDenseRestoreArrayWriteF901d(v, array, ierr)
    PetscScalar, pointer :: array(:)
    PetscErrorCode ierr
    Mat v
    call MatDenseRestoreArrayWrite(v, array, ierr)
  End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseGetArrayF902d
#endif
  Subroutine MatDenseGetArrayF902d(v, array, ierr)
    PetscScalar, pointer :: array(:, :)
    PetscErrorCode ierr
    Mat v
    call MatDenseGetArray(v, array, ierr)
  End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseRestoreArrayF902d
#endif
  Subroutine MatDenseRestoreArrayF902d(v, array, ierr)
    PetscScalar, pointer :: array(:, :)
    PetscErrorCode ierr
    Mat v
    call MatDenseRestoreArray(v, array, ierr)
  End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseGetArrayReadF902d
#endif
  Subroutine MatDenseGetArrayReadF902d(v, array, ierr)
    PetscScalar, pointer :: array(:, :)
    PetscErrorCode ierr
    Mat v
    call MatDenseGetArrayRead(v, array, ierr)
  End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseRestoreArrayReadF90
#endif
  Subroutine MatDenseRestoreArrayReadF902d(v, array, ierr)
    PetscScalar, pointer :: array(:, :)
    PetscErrorCode ierr
    Mat v
    call MatDenseRestoreArrayRead(v, array, ierr)
  End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseGetArrayWriteF90
#endif
  Subroutine MatDenseGetArrayWriteF902d(v, array, ierr)
    PetscScalar, pointer :: array(:, :)
    PetscErrorCode ierr
    Mat v
    call MatDenseGetArrayWrite(v, array, ierr)
  End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseRestoreArrayWriteF90
#endif
  Subroutine MatDenseRestoreArrayWriteF902d(v, array, ierr)
    PetscScalar, pointer :: array(:, :)
    PetscErrorCode ierr
    Mat v
    call MatDenseRestoreArrayWrite(v, array, ierr)
  End Subroutine

end module
