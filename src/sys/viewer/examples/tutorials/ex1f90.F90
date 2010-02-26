
      program ex5f90
      implicit none
#include "finclude/petscsys.h"
#include "finclude/petscviewer.h"
#include "finclude/petscviewer.h90"
      PetscViewer viewer
      PetscErrorCode ierr

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'binaryoutput',FILE_MODE_READ,viewer,ierr)
      call PetscViewerDestroy(viewer,ierr)

      call PetscFinalize(ierr)
      end
