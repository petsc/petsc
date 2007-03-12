
#include "zpetsc.h"
#include "petscda.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dagetmatrix_                 DAGETMATRIX
#define dmcompositegetentries1_      DMCOMPOSITEGETENTRIES1
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dagetmatrix_                 dagetmatrix
#define dmcompositegetentries1_      dmcompositegetentries1
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL dagetmatrix_(DA *da,CHAR mat_type PETSC_MIXED_LEN(len),Mat *J,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(mat_type,len,t);
  *ierr = DAGetMatrix(*da,t,J);
  FREECHAR(mat_type,t);
}

void PETSC_STDCALL dmcompositegetentries1_(DMComposite *dm,DA *da1)
{
  PetscErrorCode ierr = DMCompositeGetEntries(*dm,da1);
}

void PETSC_STDCALL dmcompositegetentries2_(DMComposite *dm,DA *da1,DA *da2)
{
  PetscErrorCode ierr = DMCompositeGetEntries(*dm,da1,da2);
}

void PETSC_STDCALL dmcompositegetentries3_(DMComposite *dm,DA *da1,DA *da2,DA *da3)
{
  PetscErrorCode ierr = DMCompositeGetEntries(*dm,da1,da2,da3);
}

void PETSC_STDCALL dmcompositegetentries4_(DMComposite *dm,DA *da1,DA *da2,DA *da3,DA *da4)
{
  PetscErrorCode ierr = DMCompositeGetEntries(*dm,da1,da2,da3,da4);
}

void PETSC_STDCALL dmcompositegetentries5_(DMComposite *dm,DA *da1,DA *da2,DA *da3,DA *da4,DA *da5)
{
  PetscErrorCode ierr = DMCompositeGetEntries(*dm,da1,da2,da3,da4,da5);
}
EXTERN_C_END
