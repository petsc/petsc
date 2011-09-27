#ifndef PETSC4PY_COMPAT_H
#define PETSC4PY_COMPAT_H

#include <petsc.h>
#include "compat/mpi.h"

#if PETSC_VERSION_(3,2,0)
#include "compat-32.h"
#endif

#if PETSC_VERSION_(3,1,0)
#include "compat-31.h"
#endif

#if PETSC_VERSION_(3,0,0)
#include "compat-31.h"
#endif

#endif/*PETSC4PY_COMPAT_H*/
