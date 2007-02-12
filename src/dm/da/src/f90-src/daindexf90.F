#include "petscconf.h"
#if defined(PETSC_HAVE_F90_H)
      subroutine dummy_DAGetGlobalIndicesF90()
      return
      end subroutine

#else
      Subroutine DAGetGlobalIndicesF90(v,n,array,ierr)
      implicit none
#include "include/finclude/petsc.h"
#include "include/finclude/petscda.h"
      PetscInt, pointer :: array(:)
      PetscInt  n
      PetscErrorCode ierr
      DA       v

      PetscInt,dimension(1),target :: v_local
      PetscOffset offset

      call DAGetGlobalIndices(v,n,v_local,offset,ierr)
      array=>v_local(offset+1:)
      return
      end subroutine
#endif
