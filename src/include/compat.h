#ifndef PETSC4PY_COMPAT_H
#define PETSC4PY_COMPAT_H

#include <petsc.h>
#include "compat/mpi.h"

#if PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0)
#if PETSC_VERSION_(3,0,0)
#include <petscts.h>
#include <petscda.h>
#endif
#include "compat/petsc.h"
#include "compat/petscsys.h"
#include "compat/petscfwk.h"
#include "compat/petscviewer.h"
#include "compat/petscis.h"
#include "compat/petscvec.h"
#include "compat/petscmat.h"
#include "compat/petscpc.h"
#include "compat/petscksp.h"
#include "compat/petscsnes.h"
#include "compat/petscts.h"
#include "compat/petscao.h"
#include "compat/petscdm.h"
#include "compat/petscda.h"
#include "compat/destroy.h"
#endif/*PETSC31||PETSC30*/

#endif/*PETSC4PY_COMPAT_H*/
