#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: gmres2.c,v 1.15 1999/01/13 22:56:06 curfman Exp curfman $";
#endif
#include "src/ksp/impls/gmres/gmresp.h"       /*I  "ksp.h"  I*/

#undef __FUNC__  
#define __FUNC__ "KSPGMRESSetRestart" 
/*@
    KSPGMRESSetRestart - Sets the number of search directions 
    for GMRES before restart.

    Collective on KSP

    Input Parameters:
+   ksp - the iterative context
-   max_k - the number of directions

    Level: intermediate

    Options Database Key:
.   -ksp_gmres_restart <max_k> - Sets max_k

    Note:
    The default value of max_k = 30.

.keywords: KSP, GMRES, set, restart

.seealso: KSPGMRESSetOrthogonalization(), KSPGMRESSetPreallocateVectors()
@*/
int KSPGMRESSetRestart(KSP ksp,int max_k )
{
  int ierr, (*f)(KSP,int);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)ksp,"KSPGMRESSetRestart_C",(void **)&f); CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp,max_k);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPGMRESSetOrthogonalization" 
/*@C
   KSPGMRESSetOrthogonalization - Sets the orthogonalization routine used by GMRES.

   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate
-  fcn - Orthogonalization function

   Level: intermediate

   Notes:
   Several orthogonalization routines are predefined, including

   KSPGMRESModifiedGramSchmidtOrthogonalization() - default.

   KSPGMRESUnmodifiedGramSchmidtOrthogonalization() - 
       NOT recommended; however, for some problems, particularly
       when using parallel distributed vectors, this may be
       significantly faster.

   KSPGMRESIROrthogonalization() - iterative refinement
       version of KSPGMRESUnmodifiedGramSchmidtOrthogonalization(),
       which may be more numerically stable.

   KSPGMRESDGKSOrthogonalization() - iterative refinement via the algorithm 
       by J.W. Daniel, W.B. Gragg, L. Kaufman, and G.W. Stewart,
       "Reorthogonalization and Stable Algorithms for Updating the 
       Gram-Schmidt QR Factorization", Mathematics of Computation, 
       Vol. 30, 136, 1976, pp. 772-795.  This version uses iterative
       refinement of UNMODIFIED Gram-Schmidt.  

   Options Database Keys:
+  -ksp_gmres_unmodifiedgramschmidt - Activates KSPGMRESUnmodifiedGramSchmidtOrthogonalization()
.  -ksp_gmres_irorthog - Activates KSPGMRESIROrthogonalization()
-  -ksp_gmres_dgksorthog - Activates KSPGMRESDGKSOrthogonalization() 

.keywords: KSP, GMRES, set, orthogonalization, Gram-Schmidt, iterative refinement

.seealso: KSPGMRESSetRestart(), KSPGMRESSetPreallocateVectors()
@*/
int KSPGMRESSetOrthogonalization( KSP ksp,int (*fcn)(KSP,int) )
{
  int ierr, (*f)(KSP,int (*)(KSP,int));

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)ksp,"KSPGMRESSetOrthogonalization_C",(void **)&f); CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp,fcn);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}






