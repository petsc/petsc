#include <petsc-private/fortranimpl.h>
#include <petscdm.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmgetnamedglobalvector_           DMGETNAMEDGLOBALVECTOR
#define dmrestorenamedglobalvector_       DMRESTORENAMEDGLOBALVECTOR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmgetnamedglobalvector_           dmgetnamedglobalvector
#define dmrestorenamedglobalvector_       dmrestorenamedglobalvector
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL  dmgetnamedglobalvector_(DM *dm,CHAR name PETSC_MIXED_LEN(len),Vec *X,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(name,len,t);
  *ierr = DMGetNamedGlobalVector(*dm,t,X);
  FREECHAR(name,t);
}

void PETSC_STDCALL  dmrestorenamedglobalvector_(DM *dm,CHAR name PETSC_MIXED_LEN(len),Vec *X,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(name,len,t);
  *ierr = DMRestoreNamedGlobalVector(*dm,t,X);
  FREECHAR(name,t);
}
EXTERN_C_END
