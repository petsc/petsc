/* Author:  Lisandro Dalcin   */
/* Contact: dalcinl@gmail.com */

#ifndef PETSC4PY_H
#define PETSC4PY_H

#include <Python.h>
#include <petsc.h>

#include "../../PETSc_api.h"

static int import_petsc4py(void) {
  if (import_petsc4py__PETSc() < 0) goto bad;
  return 0;
 bad:
  return -1;
}

#endif /* !PETSC4PY_H */
