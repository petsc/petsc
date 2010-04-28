/* Author:  Lisandro Dalcin   */
/* Contact: dalcinl@gmail.com */

#ifndef PETSC4PY_H
#define PETSC4PY_H

#include <Python.h>

#include <petsc.h>
#include <petscsys.h>
#include <petscis.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscksp.h>
#include <petscsnes.h>
#include <petscts.h>
#include <petscao.h>
#include <petscda.h>

#ifndef __PETSCFWK_H
struct _p_PetscFwk;
typedef struct _p_PetscFwk *PetscFwk;
#endif/*__PETSCFWK_H*/

#include "petsc4py.PETSc_api.h"

static int import_petsc4py(void) {
  if (import_petsc4py__PETSc() < 0) goto bad;
  return 0;
 bad:
  return -1;
}

#endif /* !PETSC4PY_H */
