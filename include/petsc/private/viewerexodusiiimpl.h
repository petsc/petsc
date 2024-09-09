#pragma once

#include <petscviewerexodusii.h>

#if defined(PETSC_HAVE_EXODUSII)

typedef struct {
  char         *filename;
  PetscFileMode btype;
  int           exoid;
  int           order; /* the "order" of the mesh, used to construct tri6, tetra10 cells */
  int           numNodalVariables;
  int           numZonalVariables;
  char        **nodalVariableNames;
  char        **zonalVariableNames;
} PetscViewer_ExodusII;

#endif
