
#include "private/daimpl.h"
#include "private/fortranimpl.h"
#include "petscmat.h"
#include "petscda.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define dasetLocaladiforfunction_    DASETLOCALADIFORFUNCTION
#define dasetlocaladiformffunction_  DASETLOCALADIFORMFFUNCTION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dasetlocaladiforfunction_       dasetlocaladiforfunction
#define dasetlocaladiformffunction_       dasetlocaladiformffunction
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL dasetlocaladiforfunction_(DA *da,
void (PETSC_STDCALL *jfunc)(PetscInt*,DALocalInfo*,void*,void*,PetscInt*,void*,void*,PetscInt*,void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  (*da)->adifor_lf = (DALocalFunction1)jfunc;
}

void PETSC_STDCALL dasetlocaladiformffunction_(DA *da,
void (PETSC_STDCALL *jfunc)(DALocalInfo*,void*,void*,void*,void*,void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  (*da)->adiformf_lf = (DALocalFunction1)jfunc;
}

EXTERN_C_END


