
#include <petsc-private/fortranimpl.h>
#include <petscdmda.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmdagetglobalindices_          DMDAGETGLOBALINDICES
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmdagetglobalindices_          dmdagetglobalindices
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL dmdagetglobalindices_(DM *da,PetscInt *n,PetscInt *indices,size_t *ia,PetscErrorCode *ierr)
{
  PetscInt *idx;
  *ierr = DMDAGetGlobalIndices(*da,n,&idx);
  *ia   = PetscIntAddressToFortran(indices,idx);
}

EXTERN_C_END
