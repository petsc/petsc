#include "private/fortranimpl.h"
#include "petscmat.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matgetcoloring_                  MATGETCOLORING
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matgetcoloring_                  matgetcoloring
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL matgetcoloring_(Mat *mat,CHAR type PETSC_MIXED_LEN(len),ISColoring *iscoloring,
                                   PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(type,len,t);
  *ierr = MatGetColoring(*mat,t,iscoloring);
  FREECHAR(type,t);
}

EXTERN_C_END
