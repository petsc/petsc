#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: preonly.c,v 1.20 1997/07/09 20:50:56 balay Exp bsmith $";
#endif

/*                       
       This implements a stub method that applies ONLY the preconditioner.
       This may be used in inner iterations, where it is desired to 
       allow multiple iterations as well as the "0-iteration" case
*/
#include <math.h>
#include "petsc.h"
#include "src/ksp/kspimpl.h"

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
  X    = ksp->vec_sol;
  B    = ksp->vec_rhs;
  ierr = PCApply(ksp->B,B,X); CHKERRQ(ierr);
  *its = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPCreate_PREONLY"
int KSPCreate_PREONLY(KSP ksp)
{
  PetscFunctionBegin;
  ksp->data                 = (void *) 0;
  ksp->type                 = KSPPREONLY;
  ksp->setup                = KSPSetUp_PREONLY;
  ksp->solver               = KSPSolve_PREONLY;
  ksp->adjustwork           = 0;
  ksp->destroy              = KSPDefaultDestroy;
  ksp->converged            = KSPDefaultConverged;
  ksp->buildsolution        = KSPDefaultBuildSolution;
  ksp->buildresidual        = KSPDefaultBuildResidual;
  ksp->view                 = 0;
  ksp->guess_zero           = 0; /* saves KSPSolve() unnessacarily zero x */
  PetscFunctionReturn(0);
}
