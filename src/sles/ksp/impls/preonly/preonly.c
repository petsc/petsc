#ifndef lint
static char vcid[] = "$Id: preonly.c,v 1.12 1995/10/17 21:41:12 bsmith Exp bsmith $";
#endif

/*                       
       This implements a stub method that applies ONLY the preconditioner.
       This may be used in inner iterations, where it is desired to 
       allow multiple iterations as well as the "0-iteration" case
*/
#include <stdio.h>
#include <math.h>
#include "petsc.h"
#include "kspimpl.h"

static int KSPSetUp_PREONLY(KSP itP)
{
 return KSPCheckDef( itP );
}

static int  KSPSolve_PREONLY(KSP itP,int *its)
{
  int ierr;
  Vec X,B;
  X    = itP->vec_sol;
  B    = itP->vec_rhs;
  ierr = PCApply(itP->B,B,X); CHKERRQ(ierr);
  *its = 1;
  return 0;
}

int KSPCreate_PREONLY(KSP itP)
{
  itP->data                 = (void *) 0;
  itP->type                 = KSPPREONLY;
  itP->setup                = KSPSetUp_PREONLY;
  itP->solver               = KSPSolve_PREONLY;
  itP->adjustwork           = 0;
  itP->destroy              = KSPiDefaultDestroy;
  itP->converged            = KSPDefaultConverged;
  itP->buildsolution        = KSPDefaultBuildSolution;
  itP->buildresidual        = KSPDefaultBuildResidual;
  itP->view                 = 0;
  itP->guess_zero           = 0; /* saves KSPSolve() unnessacarily zero x */
  return 0;
}
