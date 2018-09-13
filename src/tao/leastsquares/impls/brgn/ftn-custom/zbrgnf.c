#include <petsc/private/fortranimpl.h>
#include <petsc/private/f90impl.h>
#include <petsc/private/taoimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define taobrgngetsubsolver_             TAOBRGNGETSUBSOLVER
#define taobrgnsettikhonovlambda_        TAOBRGNSETTIKHONOVLAMBDA
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define taobrgngetsubsolver_             taobrgngetsubsolver
#define taobrgnsettikhonovlambda_        taobrgnsettikhonovlambda
#endif

PETSC_EXTERN void PETSC_STDCALL taobrgngetsubsolver_(Tao *tao, Tao *subsolver, PetscErrorCode *ierr)
{
    if(!*ierr) *ierr = TaoBRGNGetSubsolver(*tao, subsolver);
}

PETSC_EXTERN void PETSC_STDCALL taobrgnsettikhonovlambda_(Tao *tao, PetscReal *lambda, PetscErrorCode *ierr)
{
    if(!*ierr) *ierr = TaoBRGNSetTikhonovLambda(*tao, *lambda);
}
