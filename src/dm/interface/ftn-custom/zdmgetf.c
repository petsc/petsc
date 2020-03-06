#include <petsc/private/fortranimpl.h>
#include <petscdm.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmgetnamedglobalvector_           DMGETNAMEDGLOBALVECTOR
#define dmrestorenamedglobalvector_       DMRESTORENAMEDGLOBALVECTOR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmgetnamedglobalvector_           dmgetnamedglobalvector
#define dmrestorenamedglobalvector_       dmrestorenamedglobalvector
#endif

PETSC_EXTERN void dmgetnamedglobalvector_(DM *dm,char* name,Vec *X,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(name,len,t);
  *ierr = DMGetNamedGlobalVector(*dm,t,X);if (*ierr) return;
  FREECHAR(name,t);
}

PETSC_EXTERN void dmrestorenamedglobalvector_(DM *dm,char* name,Vec *X,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(name,len,t);
  *ierr = DMRestoreNamedGlobalVector(*dm,t,X);if (*ierr) return;
  FREECHAR(name,t);
}
