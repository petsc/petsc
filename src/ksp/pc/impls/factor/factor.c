
#include "src/ksp/pc/pcimpl.h"                /*I "petscpc.h" I*/

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

.seealso: PCFactorSetFill(), PCFactorSetShiftPd()
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
.  -pc_factor_shift [1/0] - Activate/Deactivate PCFactorSetShiftPd(); the value
   is optional with 1 being the default

   Level: intermediate

.keywords: PC, indefinite, factorization

.seealso: PCFactorSetShiftNonzero()
@*/
PetscErrorCode PCFactorSetShiftPd(PetscTruth shifting,MatFactorInfo *info)
{
  PetscFunctionBegin;
  info->shiftpd = shifting;
  PetscFunctionReturn(0);
}


