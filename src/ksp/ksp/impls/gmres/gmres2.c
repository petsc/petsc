/*$Id: gmres2.c,v 1.35 2001/08/06 21:16:44 bsmith Exp $*/
#include "src/ksp/ksp/impls/gmres/gmresp.h"       /*I  "petscksp.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "KSPGMRESSetHapTol" 
/*M
    KSPGMRESSetHapTol - Sets the tolerence for GMRES and FGMRES to declare happy breakdown.
    for GMRES before restart.

   Synopsis:
     int KSPGMRESSetHapTol(KSP ksp,PetscReal tol)

    Collective on KSP

    Input Parameters:
+   ksp - the iterative context
-   tol - the tolerance (1.e-10 is the default)

    Options Database Key:
.   -ksp_gmres_haptol <tol>

    Level: advanced

.keywords: KSP, GMRES, set, happy breakdown

.seealso: KSPGMRESSetOrthogonalization(), KSPGMRESSetPreAllocateVectors()
M*/

#undef __FUNCT__  
#define __FUNCT__ "KSPGMRESSetRestart" 
/*MC
    KSPGMRESSetRestart - Sets the number of search directions 
    for GMRES and FGMRES before restart.

   Synopsis:
     int KSPGMRESSetRestart(KSP ksp,int max_k)

    Collective on KSP

    Input Parameters:
+   ksp - the iterative context
-   max_k - the number of directions

    Options Database Key:
.   -ksp_gmres_restart <max_k> - Sets max_k

    Level: intermediate

    Note:
    The default value of max_k = 30.

.keywords: KSP, GMRES, set, restart

.seealso: KSPGMRESSetOrthogonalization(), KSPGMRESSetPreAllocateVectors()
M*/

#undef __FUNCT__  
#define __FUNCT__ "KSPGMRESSetOrthogonalization" 
/*@C
   KSPGMRESSetOrthogonalization - Sets the orthogonalization routine used by GMRES and FGMRES.

   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate
-  fcn - orthogonalization function

   Calling Sequence of function:
$   errorcode = int fcn(KSP ksp,int it);
$   it is one minus the number of GMRES iterations since last restart;
$    i.e. the size of Krylov space minus one

   Notes:
   Two orthogonalization routines are predefined, including

   KSPGMRESModifiedGramSchmidtOrthogonalization()

   KSPGMRESUnmodifiedGramSchmidtOrthogonalization() - Default. Use KSPGMRESSetUGSType() to determine if 
     iterative refinement is used to increase stability. 


   Options Database Keys:

+  -ksp_gmres_unmodifiedgramschmidt - Activates KSPGMRESUnmodifiedGramSchmidtOrthogonalization() (default)
-  -ksp_gmres_modifiedgramschmidt - Activates KSPGMRESModifiedGramSchmidtOrthogonalization()

   Level: intermediate

.keywords: KSP, GMRES, set, orthogonalization, Gram-Schmidt, iterative refinement

.seealso: KSPGMRESSetRestart(), KSPGMRESSetPreAllocateVectors(), KSPGMRESSetUGSType()
@*/
int KSPGMRESSetOrthogonalization(KSP ksp,int (*fcn)(KSP,int))
{
  int ierr,(*f)(KSP,int (*)(KSP,int));

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)ksp,"KSPGMRESSetOrthogonalization_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp,fcn);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
