
#ifndef lint
static char vcid[] = "$Id: gmpre.c,v 1.2 1997/01/27 18:15:31 bsmith Exp bsmith $";
#endif

#include "src/ksp/impls/gmres/gmresp.h"       /*I  "ksp.h"  I*/

#undef __FUNC__  
#define __FUNC__ "KSPGMRESSetPreAllocateVectors" /* ADIC Ignore */
/*@
    KSPGMRESSetPreAllocateVectors - Causes GMRES to preallocate all its
    needed work vectors at initial setup rather than the default, which 
    is to allocate them in chunks when needed.

    Input Paramter:
.   ksp   - iterative context obtained from KSPCreate

    Options Database Key:
$   -ksp_gmres_preallocate

.keywords: GMRES, preallocate, vectors

.seealso: KSPGMRESSetRestart(), KSPGMRESSetOrthogonalization()
@*/
int KSPGMRESSetPreAllocateVectors(KSP ksp)
{
  KSP_GMRES *gmres;

  if (ksp->type != KSPGMRES) return 0;
  gmres = (KSP_GMRES *)ksp->data;
  gmres->q_preallocate = 1;
  return 0;
}

