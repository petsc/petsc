
#include "src/ksp/pc/pcimpl.h"                /*I "petscpc.h" I*/
#include "src/ksp/pc/impls/factor/factor.h"

/*  Options Database Keys: ???
.  -pc_ilu_damping - add damping to diagonal to prevent zero (or very small) pivots
.  -pc_ilu_shift - apply Manteuffel shift to diagonal to force positive definite preconditioner
.  -pc_ilu_zeropivot <tol> - set tolerance for what is considered a zero pivot
 */

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetZeroPivot"
/*@
   PCFactorSetZeroPivot - Sets the size at which smaller pivots are declared to be zero

   Collective on PC
   
   Input Parameters:
+  zero - all pivots smaller than this will be considered zero
-  info - options for factorization

   Options Database Key:
.  -pc_factor_zeropivot <zero> - Sets tolerance for what is considered a zero pivot

   Level: intermediate

.keywords: PC, set, factorization, direct, fill

.seealso: PCFactorSetShiftNonzero(), PCFactorSetShiftPd()
@*/
PetscErrorCode PCFactorSetZeroPivot(PetscReal zero,MatFactorInfo *info)
{
  PetscFunctionBegin;
  info->zeropivot = zero;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetShiftNonzero"
/*@
   PCFactorSetShiftNonzero - adds this quantity to the diagonal of the matrix during 
     numerical factorization, thus the matrix has nonzero pivots

   Collective on PC
   
   Input Parameters:
+  shift - amount of shift
-  info - options for factorization

   Options Database Key:
.  -pc_factor_shiftnonzero <shift> - Sets shift amount or PETSC_DECIDE for the default

   Note: If 0.0 is given, then no shift is used. If a diagonal element is classified as a zero
         pivot, then the shift is doubled until this is alleviated.

   Level: intermediate

.keywords: PC, set, factorization, direct, fill

.seealso: PCFactorSetZeroPivot(), PCFactorSetShiftPd()
@*/
PetscErrorCode PCFactorSetShiftNonzero(PetscReal shift,MatFactorInfo *info)
{
  PetscFunctionBegin;
  if (shift == (PetscReal) PETSC_DECIDE) {
    info->shiftnz = 1.e-12;
  } else {
    info->shiftnz = shift;
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCFactorSetShiftPd"
/*@
   PCFactorSetShiftPd - specify whether to use Manteuffel shifting.
   If a matrix factorisation breaks down because of nonpositive pivots,
   adding sufficient identity to the diagonal will remedy this.
   Setting this causes a bisection method to find the minimum shift that
   will lead to a well-defined matrix factor.

   Collective on PC

   Input parameters:
+  shifting - PETSC_TRUE to set shift else PETSC_FALSE
-  info - options for factorization

   Options Database Key:
.  -pc_factor_shiftpd [1/0] - Activate/Deactivate PCFactorSetShiftPd(); the value
   is optional with 1 being the default

   Level: intermediate

.keywords: PC, indefinite, factorization

.seealso: PCFactorSetZeroPivot(), PCFactorSetShiftNonzero()
@*/
PetscErrorCode PCFactorSetShiftPd(PetscTruth shifting,MatFactorInfo *info)
{
  PetscFunctionBegin;
  info->shiftpd = shifting;
  PetscFunctionReturn(0);
}

/* shift the diagonals when zero pivot is detected */
#undef __FUNCT__  
#define __FUNCT__ "PCLUFactorCheckShift"
PetscErrorCode PCLUFactorCheckShift(Mat A,MatFactorInfo *info,Mat *B,Shift_Ctx *sctx,PetscInt *newshift)
{
  PetscReal      rs;
  PetscScalar    pv;

  PetscFunctionBegin;
  rs = sctx->rs;
  pv = sctx->pv;
  /* printf(" CheckShift: rs: %g\n",rs); */
  
  if (PetscAbsScalar(pv) <= info->zeropivot*rs && info->shiftnz){
    /* force |diag(*B)| > zeropivot*rs */
    if (!sctx->nshift){
      sctx->shift_amount = info->shiftnz;
    } else {
      sctx->shift_amount *= 2.0;
    }
    sctx->lushift = 1;
    (sctx->nshift)++;
    *newshift = PETSC_TRUE;
  } else if (PetscRealPart(pv) <= info->zeropivot*rs && info->shiftpd){ 
    /* force *B to be diagonally dominant */
    if (sctx->nshift > sctx->nshift_max) {
      SETERRQ(PETSC_ERR_CONV_FAILED,"Unable to determine shift to enforce positive definite preconditioner");
    } else if (sctx->nshift == sctx->nshift_max) {
      info->shift_fraction = sctx->shift_hi;
      sctx->lushift        = PETSC_FALSE;
    } else {
      sctx->shift_lo = info->shift_fraction; 
      info->shift_fraction = (sctx->shift_hi+sctx->shift_lo)/2.;
      sctx->lushift  = PETSC_TRUE;
    }
    sctx->shift_amount = info->shift_fraction * sctx->shift_top;
    sctx->nshift++; 
    *newshift = PETSC_TRUE;
  } else if (PetscAbsScalar(pv) <= info->zeropivot*rs){
    *newshift = -1;
  }
  PetscFunctionReturn(0);
}


