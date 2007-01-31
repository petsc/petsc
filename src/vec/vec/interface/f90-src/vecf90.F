#include "petscconf.h"
#if defined(PETSC_HAVE_F90_H)
      subroutine dummy_VecGetArrayF90()
      return
      end subroutine

#else
      subroutine VecGetArrayF90(v, array, ierr)
      implicit none

#include "include/finclude/petsc.h"
#include "include/finclude/petscvec.h"

      PetscScalar, pointer :: array(:)
      PetscErrorCode ierr
      Vec v

      PetscScalar, dimension(1), target :: v_local
      PetscOffset offset

      call VecGetArray(v,v_local,offset,ierr)
      array=>v_local(offset+1:)
      return
      end subroutine

      Subroutine VecRestoreArrayF90(v,array,ierr)
      PetscScalar, pointer :: array(:)
      PetscErrorCode ierr
      Vec     v
      PetscOffset offset

      offset = 0
      call VecRestoreArray(v,array,offset,ierr)
      nullify(array)

      return
      end subroutine

#endif
