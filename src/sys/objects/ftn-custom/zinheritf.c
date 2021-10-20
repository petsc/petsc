/*
  This file contains Fortran stubs for Options routines.
  These are not generated automatically since they require passing strings
  between Fortran and C.
*/

#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscobjectcompose_        PETSCOBJECTCOMPOSE
#define petscobjectquery_          PETSCOBJECTQUERY
#define petscobjectreference_      PETSCOBJECTREFERENCE
#define petscobjectdereference_    PETSCOBJECTDEREFERENCE
#define petscobjectgetreference_   PETSCOBJECTGETREFERENCE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscobjectcompose_        petscobjectcompose
#define petscobjectquery_          petscobjectquery
#define petscobjectreference_      petscobjectreference
#define petscobjectdereference_    petscobjectdereference
#define petscobjectgetreference_   petscobjectgetreference
#endif

/* ---------------------------------------------------------------------*/

PETSC_EXTERN void petscobjectcompose_(PetscObject *obj, char *name, PetscObject *ptr, PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *n1;

  FIXCHAR(name,len,n1);
  *ierr = PetscObjectCompose(*obj, n1, *ptr);if (*ierr) return;
  FREECHAR(name,n1);
}

PETSC_EXTERN void petscobjectquery_(PetscObject *obj, char *name, PetscObject *ptr, PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *n1;

  FIXCHAR(name,len,n1);
  *ierr = PetscObjectQuery(*obj, n1, ptr);if (*ierr) return;
  FREECHAR(name,n1);
}

PETSC_EXTERN void  petscobjectreference_(PetscObject *obj,PetscErrorCode *ierr)
{
  *ierr = PetscObjectReference(*obj);
}

PETSC_EXTERN void  petscobjectdereference_(PetscObject *obj,PetscErrorCode *ierr)
{
  *ierr = PetscObjectDereference(*obj);
}

PETSC_EXTERN void  petscobjectgetreference_(PetscObject *obj,PetscInt *ref,PetscErrorCode *ierr)
{
  *ierr = PetscObjectGetReference(*obj,ref);
}
