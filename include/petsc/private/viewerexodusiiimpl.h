#pragma once

#include <petscviewerexodusii.h>

#if defined(PETSC_HAVE_EXODUSII)

typedef struct {
  char            *filename;
  PetscFileMode    btype;
  PetscExodusIIInt exoid;
  PetscInt         order; /* the "order" of the mesh, used to construct tri6, tetra10 cells */
  PetscExodusIIInt numNodalVariables;
  PetscExodusIIInt numZonalVariables;
  char           **nodalVariableNames;
  char           **zonalVariableNames;
} PetscViewer_ExodusII;

#endif
