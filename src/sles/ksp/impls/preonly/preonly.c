/*$Id: preonly.c,v 1.31 1999/11/05 14:46:48 bsmith Exp bsmith $*/

/*                       
       This implements a stub method that applies ONLY the preconditioner.
       This may be used in inner iterations, where it is desired to 
       allow multiple iterations as well as the "0-iteration" case
*/
#include "src/sles/ksp/kspimpl.h"

#undef __FUNC__  
#define __FUNC__ "KSPSetUp_PREONLY"
static int KSPSetUp_PREONLY(KSP ksp)
{
  PetscFunctionBegin;
 PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSolve_PREONLY"
static int  KSPSolve_PREONLY(KSP ksp,int *its)
{
  int ierr;
  Vec X,B;

  PetscFunctionBegin;
  ksp->its = 0;
  X        = ksp->vec_sol;
  B        = ksp->vec_rhs;
  ierr     = KSP_PCApply(ksp,ksp->B,B,X);CHKERRQ(ierr);
  *its     = 1;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "KSPCreate_PREONLY"
int KSPCreate_PREONLY(KSP ksp)
{
  PetscFunctionBegin;
  ksp->data                      = (void *) 0;
  ksp->ops->setup                = KSPSetUp_PREONLY;
  ksp->ops->solve                = KSPSolve_PREONLY;
  ksp->ops->destroy              = KSPDefaultDestroy;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;
  ksp->ops->view                 = 0;
  ksp->guess_zero                = 0; 
  PetscFunctionReturn(0);
}
EXTERN_C_END
