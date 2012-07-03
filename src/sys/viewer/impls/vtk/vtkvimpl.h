#if !defined(_VTKVIMPL_H)
#define _VTKVIMPL_H

#include <petsc-private/viewerimpl.h>    /*I   "petscsys.h"   I*/

typedef struct _n_PetscViewerVTKObjectLink *PetscViewerVTKObjectLink;
struct _n_PetscViewerVTKObjectLink {
  PetscViewerVTKFieldType ft;
  PetscObject vec;
  PetscViewerVTKObjectLink next;
};

typedef struct {
  char                        *filename;
  PetscFileMode               btype;
  PetscBool                   written;
  PetscObject                 dm;
  PetscViewerVTKWriteFunction dmwriteall;
  PetscViewerVTKObjectLink    link;
} PetscViewer_VTK;

#endif
