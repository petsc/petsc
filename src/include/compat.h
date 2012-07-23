#ifndef PETSC4PY_COMPAT_H
#define PETSC4PY_COMPAT_H

#include <petsc.h>
#include "compat/mpi.h"

#if PETSC_VERSION_(3,2,0)
#include "compat/petsc-32.h"
#endif

#endif/*PETSC4PY_COMPAT_H*/
