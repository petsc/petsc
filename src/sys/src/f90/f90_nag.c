#ifndef lint
static char vcid[] = "$Id: zf90.c,v 1.1 1997/01/15 22:57:57 bsmith Exp bsmith $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "vec.h"
#ifdef HAVE_FORTRAN_CAPS
#define vecgetarrayf90_          VECGETARRAYF90
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define vecgetarrayf90_          vecgetarrayf90
#endif

#if defined(__cplusplus)
extern "C" {
#endif

void vecgetarrayf90_(Vec x,Scalar *fa,int *__ierr)
{
  Vec    xin = (Vec)PetscToPointer( *(int*)(x) );

  *__ierr = VecGetArray(xin,&fa); if (*__ierr) return;
}

#if defined(__cplusplus)
}
#endif
