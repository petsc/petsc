/*$Id: gmres2.c,v 1.27 2000/04/12 04:25:01 bsmith Exp balay $*/
#include "src/sles/ksp/impls/gmres/gmresp.h"       /*I  "petscksp.h"  I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"KSPGMRESSetRestart" 
/*@
    KSPGMRESSetRestart - Sets the number of search directions 
    for GMRES before restart.

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

.seealso: KSPGMRESSetOrthogonalization(), KSPGMRESSetPreallocateVectors()
@*/
int KSPGMRESSetRestart(KSP ksp,int max_k)
{
  int ierr,(*f)(KSP,int);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (max_k < 1) SETERRQ(1,1,"Restart must be positive");
  ierr = PetscObjectQueryFunction((PetscObject)ksp,"KSPGMRESSetRestart_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp,max_k);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"KSPGMRESSetOrthogonalization" 
/*@C
   KSPGMRESSetOrthogonalization - Sets the orthogonalization routine used by GMRES.

   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate
-  fcn - orthogonalization function

   Calling Sequence of function:
$   errorcode = int fcn(KSP ksp,int it);
$   it is one minus the number of GMRES iterations since last restart;
$    i.e. the size of Krylov space minus one

   Notes:
   Several orthogonalization routines are predefined, including

   KSPGMRESModifiedGramSchmidtOrthogonalization()

   KSPGMRESUnmodifiedGramSchmidtOrthogonalization() - 
       NOT recommended; however, for some problems, particularly
       when using parallel distributed vectors, this may be
       significantly faster.

   KSPGMRESIROrthogonalization() - iterative refinement
       version of KSPGMRESUnmodifiedGramSchmidtOrthogonalization(),
       which may be more numerically stable. Default

   Options Database Keys:

+  -ksp_gmres_unmodifiedgramschmidt - Activates KSPGMRESUnmodifiedGramSchmidtOrthogonalization() (default)
.  -ksp_gmres_modifiedgramschmidt - Activates KSPGMRESModifiedGramSchmidtOrthogonalization()
-  -ksp_gmres_irorthog - Activates KSPGMRESIROrthogonalization()

   Level: intermediate

.keywords: KSP, GMRES, set, orthogonalization, Gram-Schmidt, iterative refinement

.seealso: KSPGMRESSetRestart(), KSPGMRESSetPreallocateVectors()
@*/
int KSPGMRESSetOrthogonalization(KSP ksp,int (*fcn)(KSP,int))
{
  int ierr,(*f)(KSP,int (*)(KSP,int));

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)ksp,"KSPGMRESSetOrthogonalization_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp,fcn);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
