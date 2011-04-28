#include <petsc.h>
#if PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0)
#include "source/libpetsc4py.h"
#else
#include "libpetsc4py/libpetsc4py.h"
#endif
