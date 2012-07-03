
#include <petsc-private/daimpl.h>
#include <petsc-private/fortranimpl.h>
#include <petscmat.h>
#include <petscdmda.h>

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define dmdasetLocaladiforfunction_    DMDASETLOCALADIFORFUNCTION
#define dmdasetlocaladiformffunction_  DMDASETLOCALADIFORMFFUNCTION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmdasetlocaladiforfunction_       dmdasetlocaladiforfunction
#define dmdasetlocaladiformffunction_       dmdasetlocaladiformffunction
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL dmdasetlocaladiforfunction_(DM *da,void (PETSC_STDCALL *jfunc)(PetscInt*,DMDALocalInfo*,void*,void*,PetscInt*,void*,void*,PetscInt*,void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  DM_DA *dd = (DM_DA*)(*da)->data;
  (dd)->adifor_lf = (DMDALocalFunction1)jfunc;
}

void PETSC_STDCALL dmdasetlocaladiformffunction_(DM *da,void (PETSC_STDCALL *jfunc)(DMDALocalInfo*,void*,void*,void*,void*,void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  DM_DA *dd = (DM_DA*)(*da)->data;
  (dd)->adiformf_lf = (DMDALocalFunction1)jfunc;
}

EXTERN_C_END


