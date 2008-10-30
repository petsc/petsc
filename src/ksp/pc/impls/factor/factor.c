#define PETSCKSP_DLL

#include "src/ksp/pc/impls/factor/factor.h"  /*I "petscpc.h" I*/

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
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetZeroPivot(PC pc,PetscReal zero)
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
.  -pc_factor_shift_nonzero <shift> - Sets shift amount or PETSC_DECIDE for the default

   Note: If 0.0 is given, then no shift is used. If a diagonal element is classified as a zero
         pivot, then the shift is doubled until this is alleviated.

   Level: intermediate

.keywords: PC, set, factorization, direct, fill

.seealso: PCFactorSetZeroPivot(), PCFactorSetShiftPd()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetShiftNonzero(PC pc,PetscReal shift)
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
.  -pc_factor_shift_positive_definite [PETSC_TRUE/PETSC_FALSE] - Activate/Deactivate PCFactorSetShiftPd(); the value
   is optional with PETSC_TRUE being the default

   Level: intermediate

.keywords: PC, indefinite, factorization

.seealso: PCFactorSetZeroPivot(), PCFactorSetShiftNonzero()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetShiftPd(PC pc,PetscTruth shift)
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

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetUseDropTolerance"
/*@
   PCFactorSetUseDropTolerance - The preconditioner will use an ILU 
   based on a drop tolerance.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  dt - the drop tolerance, try from 1.e-10 to .1
.  dtcol - tolerance for column pivot, good values [0.1 to 0.01]
-  maxrowcount - the max number of nonzeros allowed in a row, best value
                 depends on the number of nonzeros in row of original matrix

   Options Database Key:
.  -pc_factor_use_drop_tolerance <dt,dtcol,maxrowcount> - Sets drop tolerance

   Level: intermediate

    Notes:
      This uses the iludt() code of Saad's SPARSKIT package

      There are NO default values for the 3 parameters, you must set them with reasonable values for your
      matrix. We don't know how to compute reasonable values.

.keywords: PC, levels, reordering, factorization, incomplete, ILU
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetUseDropTolerance(PC pc,PetscReal dt,PetscReal dtcol,PetscInt maxrowcount)
{
  PetscErrorCode ierr,(*f)(PC,PetscReal,PetscReal,PetscInt);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorSetUseDropTolerance_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,dt,dtcol,maxrowcount);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}  

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetLevels"
/*@
   PCFactorSetLevels - Sets the number of levels of fill to use.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  levels - number of levels of fill

   Options Database Key:
.  -pc_factor_levels <levels> - Sets fill level

   Level: intermediate

.keywords: PC, levels, fill, factorization, incomplete, ILU
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetLevels(PC pc,PetscInt levels)
{
  PetscErrorCode ierr,(*f)(PC,PetscInt);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (levels < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"negative levels");
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorSetLevels_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,levels);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetAllowDiagonalFill"
/*@
   PCFactorSetAllowDiagonalFill - Causes all diagonal matrix entries to be 
   treated as level 0 fill even if there is no non-zero location.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context

   Options Database Key:
.  -pc_factor_diagonal_fill

   Notes:
   Does not apply with 0 fill.

   Level: intermediate

.keywords: PC, levels, fill, factorization, incomplete, ILU
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetAllowDiagonalFill(PC pc)
{
  PetscErrorCode ierr,(*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorSetAllowDiagonalFill_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetShiftInBlocks"
/*@
    PCFactorSetShiftInBlocks - Shifts the diagonal of any block if LU on that block detects a zero pivot.

    Collective on PC

    Input Parameters:
+   pc - the preconditioner context
-   shift - amount to shift or PETSC_DECIDE

    Options Database Key:
.   -pc_factor_shift_in_blocks <shift>

    Level: intermediate

.seealso: PCILUSetMatOrdering(), PCFactorSetPivoting(), PCFactorSetShiftInBlocks()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetShiftInBlocks(PC pc,PetscReal shift)
{
  PetscErrorCode ierr,(*f)(PC,PetscReal);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorSetShiftInBlocks_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,shift);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorReorderForNonzeroDiagonal"
/*@
   PCFactorReorderForNonzeroDiagonal - reorders rows/columns of matrix to remove zeros from diagonal

   Collective on PC
   
   Input Parameters:
+  pc - the preconditioner context
-  tol - diagonal entries smaller than this in absolute value are considered zero

   Options Database Key:
.  -pc_factor_nonzeros_along_diagonal

   Level: intermediate

.keywords: PC, set, factorization, direct, fill

.seealso: PCFactorSetFill(), PCFactorSetShiftNonzero(), PCFactorSetZeroPivot(), MatReorderForNonzeroDiagonal()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorReorderForNonzeroDiagonal(PC pc,PetscReal rtol)
{
  PetscErrorCode ierr,(*f)(PC,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorReorderForNonzeroDiagonal_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,rtol);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetMatSolverPackage"
/*@
   PCFactorSetMatSolverPackage - sets the software that is used to perform the factorization

   Collective on PC
   
   Input Parameters:
+  pc - the preconditioner context
-  stype - for example, spooles, superlu, superlu_dist

   Options Database Key:
.  -pc_factor_mat_solver_package <stype> - spooles, petsc, superlu, superlu_dist, mumps

   Level: intermediate

   Note:
     By default this will use the PETSc factorization if it exists
     

.keywords: PC, set, factorization, direct, fill

.seealso: MatGetFactor(), MatSolverPackage, PCFactorGetMatSolverPackage()

@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetMatSolverPackage(PC pc,const MatSolverPackage stype)
{
  PetscErrorCode ierr,(*f)(PC,const MatSolverPackage);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorSetMatSolverPackage_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,stype);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorGetMatSolverPackage"
/*@
   PCFactorGetMatSolverPackage - gets the software that is used to perform the factorization

   Collective on PC
   
   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.   stype - for example, spooles, superlu, superlu_dist

   Level: intermediate


.keywords: PC, set, factorization, direct, fill

.seealso: MatGetFactor(), MatSolverPackage, PCFactorGetMatSolverPackage()

@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorGetMatSolverPackage(PC pc,const MatSolverPackage *stype)
{
  PetscErrorCode ierr,(*f)(PC,const MatSolverPackage*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorGetMatSolverPackage_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,stype);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetFill"
/*@
   PCFactorSetFill - Indicate the amount of fill you expect in the factored matrix,
   fill = number nonzeros in factor/number nonzeros in original matrix.

   Collective on PC
   
   Input Parameters:
+  pc - the preconditioner context
-  fill - amount of expected fill

   Options Database Key:
.  -pc_factor_fill <fill> - Sets fill amount

   Level: intermediate

   Note:
   For sparse matrix factorizations it is difficult to predict how much 
   fill to expect. By running with the option -info PETSc will print the 
   actual amount of fill used; allowing you to set the value accurately for
   future runs. Default PETSc uses a value of 5.0

.keywords: PC, set, factorization, direct, fill

@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetFill(PC pc,PetscReal fill)
{
  PetscErrorCode ierr,(*f)(PC,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (fill < 1.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Fill factor cannot be less then 1.0");
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorSetFill_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,fill);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetUseInPlace"
/*@
   PCFactorSetUseInPlace - Tells the system to do an in-place factorization.
   For dense matrices, this enables the solution of much larger problems. 
   For sparse matrices the factorization cannot be done truly in-place 
   so this does not save memory during the factorization, but after the matrix
   is factored, the original unfactored matrix is freed, thus recovering that
   space.

   Collective on PC

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
.  -pc_factor_in_place - Activates in-place factorization

   Notes:
   PCFactorSetUseInplace() can only be used with the KSP method KSPPREONLY or when 
   a different matrix is provided for the multiply and the preconditioner in 
   a call to KSPSetOperators().
   This is because the Krylov space methods require an application of the 
   matrix multiplication, which is not possible here because the matrix has 
   been factored in-place, replacing the original matrix.

   Level: intermediate

.keywords: PC, set, factorization, direct, inplace, in-place, LU

.seealso: PCILUSetUseInPlace()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetUseInPlace(PC pc)
{
  PetscErrorCode ierr,(*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorSetUseInPlace_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetMatOrderingType"
/*@C
    PCFactorSetMatOrderingType - Sets the ordering routine (to reduce fill) to 
    be used in the LU factorization.

    Collective on PC

    Input Parameters:
+   pc - the preconditioner context
-   ordering - the matrix ordering name, for example, MATORDERING_ND or MATORDERING_RCM

    Options Database Key:
.   -pc_factor_mat_ordering_type <nd,rcm,...> - Sets ordering routine

    Level: intermediate

    Notes: nested dissection is used by default

    For Cholesky and ICC and the SBAIJ format reorderings are not available,
    since only the upper triangular part of the matrix is stored. You can use the
    SeqAIJ format in this case to get reorderings.

@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetMatOrderingType(PC pc,const MatOrderingType ordering)
{
  PetscErrorCode ierr,(*f)(PC,const MatOrderingType);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorSetMatOrderingType_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,ordering);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetPivoting"
/*@
    PCFactorSetPivoting - Determines when pivoting is done during LU. 
      For PETSc dense matrices column pivoting is always done, for PETSc sparse matrices
      it is never done. For the Matlab and SuperLU factorization this is used.

    Collective on PC

    Input Parameters:
+   pc - the preconditioner context
-   dtcol - 0.0 implies no pivoting, 1.0 complete pivoting (slower, requires more memory but more stable)

    Options Database Key:
.   -pc_factor_pivoting <dtcol>

    Level: intermediate

.seealso: PCILUSetMatOrdering(), PCFactorSetPivotInBlocks()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetPivoting(PC pc,PetscReal dtcol)
{
  PetscErrorCode ierr,(*f)(PC,PetscReal);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorSetPivoting_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,dtcol);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetPivotInBlocks"
/*@
    PCFactorSetPivotInBlocks - Determines if pivoting is done while factoring each block
      with BAIJ or SBAIJ matrices

    Collective on PC

    Input Parameters:
+   pc - the preconditioner context
-   pivot - PETSC_TRUE or PETSC_FALSE

    Options Database Key:
.   -pc_factor_pivot_in_blocks <true,false>

    Level: intermediate

.seealso: PCILUSetMatOrdering(), PCFactorSetPivoting()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetPivotInBlocks(PC pc,PetscTruth pivot)
{
  PetscErrorCode ierr,(*f)(PC,PetscTruth);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorSetPivotInBlocks_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,pivot);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetReuseFill"
/*@
   PCFactorSetReuseFill - When matrices with same different nonzero structure are factored,
   this causes later ones to use the fill ratio computed in the initial factorization.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  flag - PETSC_TRUE to reuse else PETSC_FALSE

   Options Database Key:
.  -pc_factor_reuse_fill - Activates PCFactorSetReuseFill()

   Level: intermediate

.keywords: PC, levels, reordering, factorization, incomplete, Cholesky

.seealso: PCFactorSetReuseOrdering()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetReuseFill(PC pc,PetscTruth flag)
{
  PetscErrorCode ierr,(*f)(PC,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,2);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorSetReuseFill_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,flag);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}
