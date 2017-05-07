/*
  This file contains Fortran stubs for Options routines.
  These are not generated automatically since they require passing strings
  between Fortran and C.
*/

#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscobjectcompose_        PETSCOBJECTCOMPOSE
#define petscobjectquery_          PETSCOBJECTQUERY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscobjectcompose_        petscobjectcompose
#define petscobjectquery_          petscobjectquery
#endif

/* ---------------------------------------------------------------------*/

PETSC_EXTERN void PETSC_STDCALL petscobjectcompose_(PetscObject *obj, char *name PETSC_MIXED_LEN(len), PetscObject *ptr, PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *n1;

  FIXCHAR(name,len,n1);
  CHKFORTRANNULLOBJECTDEREFERENCE(ptr);
  *ierr = PetscObjectCompose(*obj, n1, *ptr);
  FREECHAR(name,n1);
}

PETSC_EXTERN void PETSC_STDCALL petscobjectquery_(PetscObject *obj, char *name PETSC_MIXED_LEN(len), PetscObject *ptr, PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *n1;

  FIXCHAR(name,len,n1);
  *ierr = PetscObjectQuery(*obj, n1, ptr);
  FREECHAR(name,n1);
}
