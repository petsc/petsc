#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matgetordering_                  MATGETORDERING
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matgetordering_                  matgetordering
#endif

PETSC_EXTERN void PETSC_STDCALL matgetordering_(Mat *mat,char* type PETSC_MIXED_LEN(len),IS *rperm,IS *cperm,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(type,len,t);
  *ierr = MatGetOrdering(*mat,t,rperm,cperm);
  FREECHAR(type,t);
}

