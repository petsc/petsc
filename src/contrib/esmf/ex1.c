/*$Id: ex1.c,v 1.1 2000/09/06 12:30:18 bsmith Exp bsmith $*/

#include <stdio.h>
#include "petscf90.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define cfunction CFUNCTION
#elif defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define cfunction cfunction_
#endif

#include "esmfgrid.h"


EXTERN_C_BEGIN
void cfunction_(ESMFGrid grid)
{
  
  printf("grid dimension %d",grid->dimension);
}
EXTERN_C_END
