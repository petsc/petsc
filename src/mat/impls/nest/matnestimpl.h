#ifndef PETSC_MATNESTIMPL_H
#define PETSC_MATNESTIMPL_H

#include <petsc/private/matimpl.h>

struct MatNestISPair {
  IS *row, *col;
};

typedef struct {
  PetscInt             nr, nc; /* nr x nc blocks */
  Mat                **m;
  struct MatNestISPair isglobal;
  struct MatNestISPair islocal;
  Vec                 *left, *right;
  PetscInt            *row_len, *col_len;
  PetscObjectState    *nnzstate;
  PetscBool            splitassembly;
} Mat_Nest;

#endif // PETSC_MATNESTIMPL_H
