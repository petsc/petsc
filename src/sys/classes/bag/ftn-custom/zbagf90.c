#include <petsc/private/ftnimpl.h>
#include <petscbag.h>
#include <petsc/private/bagimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscbagregisterstring_ PETSCBAGREGISTERSTRING
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscbagregisterstring_ petscbagregisterstring
#endif

PETSC_EXTERN void petscbagregisterstring_(PetscBag *bag, char *p, char *cs1, char *s1, char *s2, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T pl, PETSC_FORTRAN_CHARLEN_T cl1, PETSC_FORTRAN_CHARLEN_T l1, PETSC_FORTRAN_CHARLEN_T l2)
{
  char *t1, *t2, *ct1;
  FIXCHAR(s1, l1, t1);
  FIXCHAR(cs1, cl1, ct1);
  FIXCHAR(s2, l2, t2);
  *ierr = PetscBagRegisterString(*bag, (void *)p, (PetscInt)pl, ct1, t1, t2);
  if (*ierr) return;
  FREECHAR(cs1, ct1);
  FREECHAR(s1, t1);
  FREECHAR(s2, t2);
}
