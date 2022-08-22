#ifndef __VIEWEREXODUSIIIMPL_H
#define __VIEWEREXODUSIIIMPL_H

#include <petscviewerexodusii.h>

#if defined(PETSC_HAVE_EXODUSII)

typedef struct {
  char         *filename;
  PetscFileMode btype;
  int           exoid;
  PetscInt      order; /* the "order" of the mesh, used to construct tri6, tetra10 cells */
} PetscViewer_ExodusII;

#endif
#endif
