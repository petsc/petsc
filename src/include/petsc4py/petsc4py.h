/* Author:  Lisandro Dalcin   */
/* Contact: dalcinl@gmail.com */

#ifndef PETSC4PY_H
#define PETSC4PY_H

#include <Python.h>

#include <petsc.h>

#if PETSC_VERSION_(3,0,0)
#include <petscts.h>
#include <petscda.h>
#elif !PETSC_VERSION_(3,1,0)
#define DA DM
#endif

#ifndef __PYX_HAVE_API__petsc4py__PETSc
#include "petsc4py.PETSc_api.h"
static int import_petsc4py(void) {
  if (import_petsc4py__PETSc() < 0) goto bad;
  return 0;
 bad:
  return -1;
}
#endif

#endif /* !PETSC4PY_H */
