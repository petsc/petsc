
#include "src/ksp/pc/pcimpl.h"                /*I "petscpc.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetShiftNonzero"
/*@
   PCFactorSetShiftNonzero - adds this quantity to the diagonal of the matrix during 
     numerical factorization, thus the matrix has nonzero pivots

   Collective on PC
   
   Input Parameters:
+  shift - amount of shift
-  info - 

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
    info->damping = 1.e-12;
  } else {
    info->damping = shift;
  } 
  PetscFunctionReturn(0);
}
