
#include "private/fortranimpl.h"
#include "petscis.h"
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define islocaltoglobalmappingapply_  ISLOCALTOGLOBALMAPPINGAPPLY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define islocaltoglobalmappingapply_  islocaltoglobalmappingapply
#endif

EXTERN_C_BEGIN

/*
   This is the same as the macro ISLocalToGlobalMappingApply() except it does not
  return error codes.
*/
void PETSC_STDCALL islocaltoglobalmappingapply_(ISLocalToGlobalMapping *mapping,PetscInt *N,PetscInt *in,PetscInt *out,PetscErrorCode *ierr)
{
  PetscInt i,*idx = (*mapping)->indices,Nmax = (*mapping)->n;
  for (i=0; i<(*N); i++) {
    if (in[i] < 0) {out[i] = in[i]; continue;}
    if (in[i] >= Nmax) {
      *ierr = PetscError(__LINE__,"ISLocalToGlobalMappingApply_Fortran",__FILE__,__SDIR__,1,1,"Index out of range");
      return;
    }
    out[i] = idx[in[i]];
  }
}

EXTERN_C_END

