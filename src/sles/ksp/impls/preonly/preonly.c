#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: preonly.c,v 1.28 1999/01/31 16:09:01 bsmith Exp balay $";
#endif

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
  ierr     = PCApply(ksp->B,B,X);CHKERRQ(ierr);
  *its     = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSolveTrans_PREONLY"
static int  KSPSolveTrans_PREONLY(KSP ksp,int *its)
{
  int ierr;
  Vec X,B;

  PetscFunctionBegin;
  ksp->its = 0;
  X        = ksp->vec_sol;
  B        = ksp->vec_rhs;
  ierr     = PCApplyTrans(ksp->B,B,X);CHKERRQ(ierr);
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
  ksp->ops->solvetrans           = KSPSolveTrans_PREONLY;
  ksp->ops->destroy              = KSPDefaultDestroy;
  ksp->converged                 = KSPDefaultConverged;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;
  ksp->ops->view                 = 0;
  ksp->guess_zero                = 0; 
  PetscFunctionReturn(0);
}
EXTERN_C_END
