#ifndef lint
static char vcid[] = "$Id: preonly.c,v 1.16 1996/04/05 05:58:05 bsmith Exp bsmith $";
#endif

/*                       
       This implements a stub method that applies ONLY the preconditioner.
       This may be used in inner iterations, where it is desired to 
       allow multiple iterations as well as the "0-iteration" case
*/
#include <stdio.h>
#include <math.h>
#include "petsc.h"
#include "src/ksp/kspimpl.h"

static int KSPSetUp_PREONLY(KSP ksp)
{
 return 0;
}

static int  KSPSolve_PREONLY(KSP ksp,int *its)
{
  int ierr;
  Vec X,B;
  X    = ksp->vec_sol;
  B    = ksp->vec_rhs;
  ierr = PCApply(ksp->B,B,X); CHKERRQ(ierr);
  *its = 1;
  return 0;
}

int KSPCreate_PREONLY(KSP ksp)
{
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
  return 0;
}
