#ifndef lint
static char vcid[] = "$Id: cgtype.c,v 1.2 1997/01/27 18:15:28 bsmith Exp bsmith $";
#endif

#include "src/ksp/impls/cg/cgctx.h"       /*I "ksp.h" I*/

#undef __FUNC__  
#define __FUNC__ "KSPCGSetType" /* ADIC Ignore */
/*@
    KSPCGSetType - Sets the variant of the conjugate gradient method to
    use for solving a linear system with a complex coefficient matrix.
    This option is irrelevant when solving a real system.

    Input Parameters:
.   ksp - the iterative context
.   type - the variant of CG to use, one of
$     KSP_CG_HERMITIAN - complex, Hermitian matrix (default)
$     KSP_CG_SYMMETRIC - complex, symmetric matrix

    Options Database Keys:
$   -ksp_cg_Hermitian
$   -ksp_cg_symmetric

    Note:
    By default, the matrix is assumed to be complex, Hermitian.

.keywords: CG, conjugate gradient, Hermitian, symmetric, set, type
@*/
int KSPCGSetType(KSP ksp,KSPCGType type)
{
  KSP_CG *cg;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  cg = (KSP_CG *)ksp->data;
  if (ksp->type != KSPCG) return 0;
  cg->type = type;
  return 0;
}





