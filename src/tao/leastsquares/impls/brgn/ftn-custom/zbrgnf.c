#include <petsc/private/fortranimpl.h>
#include <petsc/private/f90impl.h>
#include <petsc/private/taoimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define taobrgngetsubsolver_             TAOBRGNGETSUBSOLVER
#define taobrgnsettikhonovlambda_        TAOBRGNSETTIKHONOVLAMBDA
#define taobrgnsetl1smoothepsilon_       TAOBRGNSETL1SMOOTHEPSILON
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define taobrgngetsubsolver_             taobrgngetsubsolver
#define taobrgnsettikhonovlambda_        taobrgnsettikhonovlambda
#define taobrgnsetl1smoothepsilon_       taobrgnsetl1smoothepsilon
#endif

PETSC_EXTERN void PETSC_STDCALL taobrgngetsubsolver_(Tao *tao, Tao *subsolver, PetscErrorCode *ierr)
{
    if(!*ierr) *ierr = TaoBRGNGetSubsolver(*tao, subsolver);
}

PETSC_EXTERN void PETSC_STDCALL taobrgnsettikhonovlambda_(Tao *tao, PetscReal *lambda, PetscErrorCode *ierr)
{
    if(!*ierr) *ierr = TaoBRGNSetTikhonovLambda(*tao, *lambda);
}

PETSC_EXTERN void PETSC_STDCALL taobrgnsetl1smoothepsilon_(Tao *tao, PetscReal *epsilon, PetscErrorCode *ierr)
{
    if(!*ierr) *ierr = TaoBRGNSetL1SmoothEpsilon(*tao, *epsilon);
}

