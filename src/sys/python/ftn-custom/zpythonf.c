#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscpythoninitialize_ PETSCPYTHONINITIALIZE
#define petscpythonfinalize_   PETSCPYTHONFINALIZE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscpythoninitialize_ petscpythoninitialize
#define petscpythonfinalize_   petscpythonfinalize
#endif

PETSC_EXTERN void PETSC_STDCALL petscpythoninitialize_(char* n1 PETSC_MIXED_LEN(l1),char* n2 PETSC_MIXED_LEN(l2), PetscErrorCode *ierr PETSC_END_LEN(l1) PETSC_END_LEN(l2))
{
  char *t1,*t2;
  FIXCHAR(n1,l1,t1);
  FIXCHAR(n2,l2,t2);
  *ierr = PetscPythonInitialize(t1,t2);if (*ierr) return;
  FREECHAR(n1,t1);
  FREECHAR(n2,t2);
}

PETSC_EXTERN void PETSC_STDCALL petscpythonfinalize_(PetscErrorCode *ierr)
{
  *ierr = PetscPythonFinalize();
}
