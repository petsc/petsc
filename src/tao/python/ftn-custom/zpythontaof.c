#include <petsc/private/fortranimpl.h>
#include <petsctao.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define taopythonsettype_            TAOPYTHONSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define taopythonsettype_            taopythonsettype
#endif

PETSC_EXTERN void taopythonsettype_(Tao *tao, char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(name,len,t);
  *ierr = TaoPythonSetType(*tao,t);if (*ierr) return;
  FREECHAR(name,t);
}
