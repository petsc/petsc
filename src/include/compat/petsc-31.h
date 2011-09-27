#ifndef PETSC4PY_COMPAT_31_H
#define PETSC4PY_COMPAT_31_H

#if PETSC_VERSION_(3,0,0)
#include <petscts.h>
#include <petscda.h>
#endif

#include "petsc-31/petsc.h"
#include "petsc-31/petscsys.h"
#include "petsc-31/petscfwk.h"
#include "petsc-31/petscviewer.h"
#include "petsc-31/petscis.h"
#include "petsc-31/petscvec.h"
#include "petsc-31/petscmat.h"
#include "petsc-31/petscpc.h"
#include "petsc-31/petscksp.h"
#include "petsc-31/petscsnes.h"
#include "petsc-31/petscts.h"
#include "petsc-31/petscao.h"
#include "petsc-31/petscdm.h"
#include "petsc-31/petscda.h"
#include "petsc-31/destroy.h"

#endif/*PETSC4PY_COMPAT_31_H*/
