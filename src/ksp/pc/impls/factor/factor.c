
#include "src/ksp/pc/pcimpl.h"                /*I "petscpc.h" I*/

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
+  pc - the preconditioner context
-  zero - all pivots smaller than this will be considered zero

   Options Database Key:
.  -pc_factor_zeropivot <zero> - Sets tolerance for what is considered a zero pivot

   Level: intermediate

.keywords: PC, set, factorization, direct, fill

.seealso: PCFactorSetShiftNonzero(), PCFactorSetShiftPd()
@*/
PetscErrorCode PCFactorSetZeroPivot(PC pc,PetscReal zero)
{
  PetscErrorCode ierr,(*f)(PC,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorSetZeroPivot_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,zero);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetShiftNonzero"
/*@
   PCFactorSetShiftNonzero - adds this quantity to the diagonal of the matrix during 
     numerical factorization, thus the matrix has nonzero pivots

   Collective on PC
   
   Input Parameters:
+  pc - the preconditioner context
-  shift - amount of shift

   Options Database Key:
.  -pc_factor_shiftnonzero <shift> - Sets shift amount or PETSC_DECIDE for the default

   Note: If 0.0 is given, then no shift is used. If a diagonal element is classified as a zero
         pivot, then the shift is doubled until this is alleviated.

   Level: intermediate

.keywords: PC, set, factorization, direct, fill

.seealso: PCFactorSetZeroPivot(), PCFactorSetShiftPd()
@*/
PetscErrorCode PCFactorSetShiftNonzero(PC pc,PetscReal shift)
{
  PetscErrorCode ierr,(*f)(PC,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorSetShiftNonzero_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,shift);CHKERRQ(ierr);
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
+  pc - the preconditioner context
-  shifting - PETSC_TRUE to set shift else PETSC_FALSE

   Options Database Key:
.  -pc_factor_shiftpd [1/0] - Activate/Deactivate PCFactorSetShiftPd(); the value
   is optional with 1 being the default

   Level: intermediate

.keywords: PC, indefinite, factorization

.seealso: PCFactorSetZeroPivot(), PCFactorSetShiftNonzero()
@*/
PetscErrorCode PCFactorSetShiftPd(PC pc,PetscTruth shift)
{
  PetscErrorCode ierr,(*f)(PC,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorSetShiftPd_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,shift);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}
