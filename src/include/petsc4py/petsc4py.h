#ifndef PETSC4PY_H
#define PETSC4PY_H

#include <Python.h>

#include <petsc.h>
#include <petscis.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscksp.h>
#include <petscsnes.h>
#include <petscts.h>
#include <petscao.h>
#include <petscda.h>

#include "petsc4py_PETSc_api.h"

static int import_petsc4py(void) {
  if (import_petsc4py__PETSc() < 0) goto bad;
  return 0;
 bad:
  return -1;
}

#endif /* !PETSC4PY_H */
