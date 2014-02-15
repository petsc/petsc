#if !defined(__TAODEF_H)
#define __TAODEF_H

#include "finclude/petsctsdef.h"

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define Tao PetscFortranAddr
#define TaoLineSearch PetscFortranAddr
#define TaoTerminationReason integer
#endif

#endif
