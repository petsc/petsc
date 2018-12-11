#include <petsc/private/fortranimpl.h>
#include <petsc/private/f90impl.h>
#include <petsc/private/taoimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define taobrgngetsubsolver_             TAOBRGNGETSUBSOLVER
#define taobrgnsetl1regularizerweight_   TAOBRGNSETL1REGULARIZERWEIGHT
#define taobrgnsetl1smoothepsilon_       TAOBRGNSETL1SMOOTHEPSILON
#define taobrgnsetdictionarymatrix_      TAOBRGNSETDICTIONARYMATRIX
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define taobrgngetsubsolver_             taobrgngetsubsolver
#define taobrgnsetl1regularizerweight_   taobrgnsetl1regularizerweight
#define taobrgnsetl1smoothepsilon_       taobrgnsetl1smoothepsilon
#define taobrgnsetdictionarymatrix_      taobrgnsetdictionarymatrix
#endif

PETSC_EXTERN void PETSC_STDCALL taobrgngetsubsolver_(Tao *tao, Tao *subsolver, PetscErrorCode *ierr)
{
    if(!*ierr) *ierr = TaoBRGNGetSubsolver(*tao, subsolver);
}

PETSC_EXTERN void PETSC_STDCALL taobrgnsetl1regularizerweight_(Tao *tao, PetscReal *lambda, PetscErrorCode *ierr)
{
    if(!*ierr) *ierr = TaoBRGNSetL1RegularizerWeight(*tao, *lambda);
}

PETSC_EXTERN void PETSC_STDCALL taobrgnsetl1smoothepsilon_(Tao *tao, PetscReal *epsilon, PetscErrorCode *ierr)
{
    if(!*ierr) *ierr = TaoBRGNSetL1SmoothEpsilon(*tao, *epsilon);
}

PETSC_EXTERN void PETSC_STDCALL taobrgnsetdictionarymatrix(Tao *tao, Mat *dict, PetscErrorCode *ierr)
{
    if(!*ierr) *ierr = TaoBRGNSetDictionaryMatrix(*tao, *dict);
}
