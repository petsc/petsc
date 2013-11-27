#if !defined(__TAOSOLVERDEF_H)
#define __TAOSOLVERDEF_H

#include "finclude/petsctsdef.h"

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define TaoSolver PetscFortranAddr
#define TaoLineSearch PetscFortranAddr
#define TaoSolverTerminationReason integer
#endif

#endif
