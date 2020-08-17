#include <petsc/private/fortranimpl.h>

/*
   We need this stub function in a separate file that does not include petscviewer.h so that PETSc Fortran
   builds do not print messages about deprecated functions
*/
typedef PetscEnum PetscViewerFormat;

PETSC_EXTERN PetscErrorCode PetscViewerSetFormat(PetscViewer,PetscViewerFormat);

PETSC_EXTERN PetscErrorCode PetscViewerSetFormatDeprecated(PetscViewer v,PetscViewerFormat f)
{
  return PetscViewerSetFormat(v,f);
}
