
#include "src/mat/matimpl.h"  /*I   "petscmat.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "Mat_LUFactorCheckShift"
/*@C
   Mat_LUFactorCheckShift -  shift the diagonals when zero pivot is detected 

   Collective on Mat

   Input Parameters:
+  info - information about the matrix factorization 
.  sctx - pointer to the struct LUShift_Ctx
-  newshift - 0: shift is unchanged; 1: shft is updated; -1: zeropivot  

   Level: developer
@*/
PetscErrorCode Mat_LUFactorCheckShift(MatFactorInfo *info,LUShift_Ctx *sctx,PetscInt *newshift)
{
  PetscReal      rs = sctx->rs;
  PetscScalar    pv = sctx->pv;

  PetscFunctionBegin;
  if (PetscAbsScalar(pv) <= info->zeropivot*rs && info->shiftnz){
    /* force |diag| > zeropivot*rs */
    if (!sctx->nshift){
      sctx->shift_amount = info->shiftnz;
    } else {
      sctx->shift_amount *= 2.0;
    }
    sctx->lushift = PETSC_TRUE;
    (sctx->nshift)++;
    *newshift = PETSC_TRUE;
  } else if (PetscRealPart(pv) <= info->zeropivot*rs && info->shiftpd){ 
    /* force matfactor to be diagonally dominant */
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
