#include <petsc/private/fortranimpl.h>
#include "petscsys.h"
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscgetarchtype_                  PETSCGETARCHTYPE
#define petscbarrier_                      PETSCBARRIER
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscgetarchtype_                  petscgetarchtype
#define petscbarrier_                      petscbarrier
#endif

PETSC_EXTERN void PETSC_STDCALL petscgetarchtype_(char* str PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char   *tstr;
  size_t tlen;
  tstr  = str;
  tlen  = len; /* int to size_t */
  *ierr = PetscGetArchType(tstr,tlen);
  FIXRETURNCHAR(PETSC_TRUE,str,len);
}

PETSC_EXTERN void PETSC_STDCALL  petscbarrier_(PetscObject *obj, int *ierr){
  *ierr = PetscBarrier(*obj);
}

