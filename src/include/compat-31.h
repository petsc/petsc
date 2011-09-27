#ifndef PETSC4PY_COMPAT_31_H
#define PETSC4PY_COMPAT_31_H

#if PETSC_VERSION_(3,0,0)
#include <petscts.h>
#include <petscda.h>
#endif

#include "compat-31/petsc.h"
#include "compat-31/petscsys.h"
#include "compat-31/petscfwk.h"
#include "compat-31/petscviewer.h"
#include "compat-31/petscis.h"
#include "compat-31/petscvec.h"
#include "compat-31/petscmat.h"
#include "compat-31/petscpc.h"
#include "compat-31/petscksp.h"
#include "compat-31/petscsnes.h"
#include "compat-31/petscts.h"
#include "compat-31/petscao.h"
#include "compat-31/petscdm.h"
#include "compat-31/petscda.h"
#include "compat-31/destroy.h"

#endif/*PETSC4PY_COMPAT_31_H*/
