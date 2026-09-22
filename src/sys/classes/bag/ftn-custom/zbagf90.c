#include <petsc/private/ftnimpl.h>
#include <petscbag.h>
#include <petsc/private/bagimpl.h>
#include <petscviewer.h>

#if PetscDefined(HAVE_FORTRAN_CAPS)
  #define petscbagregisterstring_ PETSCBAGREGISTERSTRING
#elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
  #define petscbagregisterstring_ petscbagregisterstring
#endif

PETSC_EXTERN void petscbagregisterstring_(PetscBag *bag, char *addr, char *mdefault, char *name, char *help, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T l_addr, PETSC_FORTRAN_CHARLEN_T l_mdefault, PETSC_FORTRAN_CHARLEN_T l_name, PETSC_FORTRAN_CHARLEN_T l_help)
{
  char *t1, *t2, *ct1;
  FIXCHAR(name, l_name, t1);
  FIXCHAR(mdefault, l_mdefault, ct1);
  FIXCHAR(help, l_help, t2);
  *ierr = PetscBagRegisterString(*bag, (void *)addr, (PetscInt)l_addr, ct1, t1, t2);
  if (*ierr) return;
  FREECHAR(mdefault, ct1);
  FREECHAR(name, t1);
  FREECHAR(help, t2);
}
