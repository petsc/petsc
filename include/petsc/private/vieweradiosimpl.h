#ifndef PETSC_VIEWER_ADIOS_IMPL_H
#define PETSC_VIEWER_ADIOS_IMPL_H

typedef struct {
  char         *filename;
  PetscFileMode btype;
  PetscInt      timestep;
  int64_t       adios_handle;
  ADIOS_FILE   *adios_fp;
} PetscViewer_ADIOS;

#endif
