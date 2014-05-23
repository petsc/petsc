
#include <petsc-private/fortranimpl.h>
#include <petscdmda.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmdagetglobalindices_          DMDAGETGLOBALINDICES
#define dmdarestoreglobalindices_      DMDARESTOREGLOBALINDICES
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmdagetglobalindices_          dmdagetglobalindices
#define dmdarestoreglobalindices_      dmdarestoreglobalindices
#endif

PETSC_EXTERN void PETSC_STDCALL dmdagetglobalindices_(DM *da,PetscInt *n,PetscInt *indices,size_t *ia,PetscErrorCode *ierr)
{
  const PetscInt *idx;
  *ierr = DMDAGetGlobalIndices(*da,n,&idx);
  *ia   = PetscIntAddressToFortran(indices,idx);
}

PETSC_EXTERN void PETSC_STDCALL dmdarestoreglobalindices_(DM *da,PetscInt *n,PetscInt *fa,size_t *ia,PetscErrorCode *ierr)
{
  const PetscInt *lx = PetscIntAddressFromFortran(fa,*ia);
  *ierr = DMDARestoreGlobalIndices(*da,n,&lx);
}

