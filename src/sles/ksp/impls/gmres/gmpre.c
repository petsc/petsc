#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: gmpre.c,v 1.19 1999/01/31 21:30:45 curfman Exp balay $";
#endif

#include "src/sles/ksp/impls/gmres/gmresp.h"       /*I  "ksp.h"  I*/

#undef __FUNC__  
#define __FUNC__ "KSPGMRESSetPreAllocateVectors" 
/*@
    KSPGMRESSetPreAllocateVectors - Causes GMRES to preallocate all its
    needed work vectors at initial setup rather than the default, which 
    is to allocate them in chunks when needed.

    Collective on KSP

    Input Parameter:
.   ksp   - iterative context obtained from KSPCreate

    Options Database Key:
.   -ksp_gmres_preallocate - Activates KSPGmresSetPreAllocateVectors()

    Level: intermediate

.keywords: GMRES, preallocate, vectors

.seealso: KSPGMRESSetRestart(), KSPGMRESSetOrthogonalization()
@*/
int KSPGMRESSetPreAllocateVectors(KSP ksp)
{
  int ierr, (*f)(KSP);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)ksp,"KSPGMRESSetPreAllocateVectors_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPGMRESPrestartSet" 
/*@
    KSPGMRESPrestartSet - Sets the number of vectors that GMRES will reuse in 
    future solves from the first solver after this call.

    Collective on KSP

    Input Parameter:
+   ksp   - iterative context obtained from KSPCreate
-   pre - number of directions

    Options Database Key:
.   -ksp_gmres_prestart <pre>

    Level: intermediate

.keywords: KSP, GMRES, vectors prestarted GMRES

.seealso: KSPGMRESSetRestart(), KSPGMRESSetOrthogonalization()
@*/
int KSPGMRESPrestartSet(KSP ksp,int pre)
{
  int ierr, (*f)(KSP,int);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)ksp,"KSPGMRESPrestartSet_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp,pre);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
