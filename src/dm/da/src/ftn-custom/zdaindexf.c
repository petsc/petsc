
#include "private/fortranimpl.h"
#include "petscda.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dagetglobalindices_          DAGETGLOBALINDICES
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dagetglobalindices_          dagetglobalindices
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL dagetglobalindices_(DA *da,PetscInt *n,PetscInt *indices,size_t *ia,PetscErrorCode *ierr)
{
  PetscInt *idx;
  *ierr = DAGetGlobalIndices(*da,n,&idx);
  *ia   = PetscIntAddressToFortran(indices,idx);
}

EXTERN_C_END
