#ifndef __VIEWEREXODUSIIIMPL_H
#define __VIEWEREXODUSIIIMPL_H

#include <petscviewerexodusii.h>

#if defined(PETSC_HAVE_EXODUSII)

typedef struct {
  char         *filename;
  PetscFileMode btype;
  int           exoid;
} PetscViewer_ExodusII;

#endif
#endif
