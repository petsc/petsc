
#include <../src/ksp/pc/impls/factor/factor.h>  /*I "petscpc.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetUpMatSolverPackage"
/*@
    PCFactorSetUpMatSolverPackage - Can be called after KSPSetOperators() or PCSetOperators(), causes MatGetFactor() to be called so then one may 
       set the options for that particular factorization object.

  Input Parameter:
.  pc  - the preconditioner context

  Notes: After you have called this function (which has to be after the KSPSetOperators() or PCSetOperators()) you can call PCFactorGetMatrix() and then set factor options on that matrix.

.seealso: PCFactorSetMatSolverPackage(), PCFactorGetMatrix()

  Level: intermediate

@*/
PetscErrorCode PCFactorSetUpMatSolverPackage(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCFactorSetUpMatSolverPackage_C",(PC),(pc));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetZeroPivot"
/*@
   PCFactorSetZeroPivot - Sets the size at which smaller pivots are declared to be zero

   Logically Collective on PC
   
   Input Parameters:
+  pc - the preconditioner context
-  zero - all pivots smaller than this will be considered zero

   Options Database Key:
.  -pc_factor_zeropivot <zero> - Sets tolerance for what is considered a zero pivot

   Level: intermediate

.keywords: PC, set, factorization, direct, fill

.seealso: PCFactorSetShiftType(), PCFactorSetShiftAmount()
@*/
PetscErrorCode  PCFactorSetZeroPivot(PC pc,PetscReal zero)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveReal(pc,zero,2);
  ierr = PetscTryMethod(pc,"PCFactorSetZeroPivot_C",(PC,PetscReal),(pc,zero));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetShiftType"
/*@
   PCFactorSetShiftType - adds a particular type of quantity to the diagonal of the matrix during 
     numerical factorization, thus the matrix has nonzero pivots

   Logically Collective on PC
   
   Input Parameters:
+  pc - the preconditioner context
-  shifttype - type of shift; one of MAT_SHIFT_NONE, MAT_SHIFT_NONZERO,  MAT_SHIFT_POSITIVE_DEFINITE, MAT_SHIFT_INBLOCKS 

   Options Database Key:
.  -pc_factor_shift_type <shifttype> - Sets shift type or PETSC_DECIDE for the default; use '-help' for a list of available types

   Level: intermediate

.keywords: PC, set, factorization, 

.seealso: PCFactorSetZeroPivot(), PCFactorSetShiftAmount()
@*/
PetscErrorCode  PCFactorSetShiftType(PC pc,MatFactorShiftType shifttype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveEnum(pc,shifttype,2);
  ierr = PetscTryMethod(pc,"PCFactorSetShiftType_C",(PC,MatFactorShiftType),(pc,shifttype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetShiftAmount"
/*@
   PCFactorSetShiftAmount - adds a quantity to the diagonal of the matrix during 
     numerical factorization, thus the matrix has nonzero pivots

   Logically Collective on PC
   
   Input Parameters:
+  pc - the preconditioner context
-  shiftamount - amount of shift 

   Options Database Key:
.  -pc_factor_shift_amount <shiftamount> - Sets shift amount or PETSC_DECIDE for the default

   Level: intermediate

.keywords: PC, set, factorization, 

.seealso: PCFactorSetZeroPivot(), PCFactorSetShiftType()
@*/
PetscErrorCode  PCFactorSetShiftAmount(PC pc,PetscReal shiftamount)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveReal(pc,shiftamount,2);
  ierr = PetscTryMethod(pc,"PCFactorSetShiftAmount_C",(PC,PetscReal),(pc,shiftamount));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetDropTolerance"
/*
   PCFactorSetDropTolerance - The preconditioner will use an ILU 
   based on a drop tolerance. (Under development)

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  dt - the drop tolerance, try from 1.e-10 to .1
.  dtcol - tolerance for column pivot, good values [0.1 to 0.01]
-  maxrowcount - the max number of nonzeros allowed in a row, best value
                 depends on the number of nonzeros in row of original matrix

   Options Database Key:
.  -pc_factor_drop_tolerance <dt,dtcol,maxrowcount> - Sets drop tolerance

   Level: intermediate

      There are NO default values for the 3 parameters, you must set them with reasonable values for your
      matrix. We don't know how to compute reasonable values.

.keywords: PC, levels, reordering, factorization, incomplete, ILU
*/
PetscErrorCode  PCFactorSetDropTolerance(PC pc,PetscReal dt,PetscReal dtcol,PetscInt maxrowcount)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveReal(pc,dtcol,2);
  PetscValidLogicalCollectiveInt(pc,maxrowcount,3);
  ierr = PetscTryMethod(pc,"PCFactorSetDropTolerance_C",(PC,PetscReal,PetscReal,PetscInt),(pc,dt,dtcol,maxrowcount));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}  

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetLevels"
/*@
   PCFactorSetLevels - Sets the number of levels of fill to use.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  levels - number of levels of fill

   Options Database Key:
.  -pc_factor_levels <levels> - Sets fill level

   Level: intermediate

.keywords: PC, levels, fill, factorization, incomplete, ILU
@*/
PetscErrorCode  PCFactorSetLevels(PC pc,PetscInt levels)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (levels < 0) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"negative levels");
  PetscValidLogicalCollectiveInt(pc,levels,2);
  ierr = PetscTryMethod(pc,"PCFactorSetLevels_C",(PC,PetscInt),(pc,levels));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetAllowDiagonalFill"
/*@
   PCFactorSetAllowDiagonalFill - Causes all diagonal matrix entries to be 
   treated as level 0 fill even if there is no non-zero location.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context

   Options Database Key:
.  -pc_factor_diagonal_fill

   Notes:
   Does not apply with 0 fill.

   Level: intermediate

.keywords: PC, levels, fill, factorization, incomplete, ILU
@*/
PetscErrorCode  PCFactorSetAllowDiagonalFill(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCFactorSetAllowDiagonalFill_C",(PC),(pc));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorReorderForNonzeroDiagonal"
/*@
   PCFactorReorderForNonzeroDiagonal - reorders rows/columns of matrix to remove zeros from diagonal

   Logically Collective on PC
   
   Input Parameters:
+  pc - the preconditioner context
-  tol - diagonal entries smaller than this in absolute value are considered zero

   Options Database Key:
.  -pc_factor_nonzeros_along_diagonal

   Level: intermediate

.keywords: PC, set, factorization, direct, fill

.seealso: PCFactorSetFill(), PCFactorSetShiftNonzero(), PCFactorSetZeroPivot(), MatReorderForNonzeroDiagonal()
@*/
PetscErrorCode  PCFactorReorderForNonzeroDiagonal(PC pc,PetscReal rtol)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveReal(pc,rtol,2);
  ierr = PetscTryMethod(pc,"PCFactorReorderForNonzeroDiagonal_C",(PC,PetscReal),(pc,rtol));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetMatSolverPackage"
/*@C
   PCFactorSetMatSolverPackage - sets the software that is used to perform the factorization

   Logically Collective on PC
   
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
PetscErrorCode  PCFactorSetMatSolverPackage(PC pc,const MatSolverPackage stype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCFactorSetMatSolverPackage_C",(PC,const MatSolverPackage),(pc,stype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorGetMatSolverPackage"
/*@C
   PCFactorGetMatSolverPackage - gets the software that is used to perform the factorization

   Not Collective
   
   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.   stype - for example, spooles, superlu, superlu_dist (PETSC_NULL if the PC does not have a solver package)

   Level: intermediate


.keywords: PC, set, factorization, direct, fill

.seealso: MatGetFactor(), MatSolverPackage, PCFactorGetMatSolverPackage()

@*/
PetscErrorCode  PCFactorGetMatSolverPackage(PC pc,const MatSolverPackage *stype)
{
  PetscErrorCode ierr,(*f)(PC,const MatSolverPackage*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorGetMatSolverPackage_C",(void(**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,stype);CHKERRQ(ierr);
  } else {
    *stype = PETSC_NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetFill"
/*@
   PCFactorSetFill - Indicate the amount of fill you expect in the factored matrix,
   fill = number nonzeros in factor/number nonzeros in original matrix.

   Not Collective, each process can expect a different amount of fill
   
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

   This parameter has NOTHING to do with the levels-of-fill of ILU(). That is set with PCFactorSetLevels() or -pc_factor_levels.
    

.keywords: PC, set, factorization, direct, fill

@*/
PetscErrorCode  PCFactorSetFill(PC pc,PetscReal fill)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (fill < 1.0) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Fill factor cannot be less then 1.0");
  ierr = PetscTryMethod(pc,"PCFactorSetFill_C",(PC,PetscReal),(pc,fill));CHKERRQ(ierr);
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

   Logically Collective on PC

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
PetscErrorCode  PCFactorSetUseInPlace(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCFactorSetUseInPlace_C",(PC),(pc));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetMatOrderingType"
/*@C
    PCFactorSetMatOrderingType - Sets the ordering routine (to reduce fill) to 
    be used in the LU factorization.

    Logically Collective on PC

    Input Parameters:
+   pc - the preconditioner context
-   ordering - the matrix ordering name, for example, MATORDERINGND or MATORDERINGRCM

    Options Database Key:
.   -pc_factor_mat_ordering_type <nd,rcm,...> - Sets ordering routine

    Level: intermediate

    Notes: nested dissection is used by default

    For Cholesky and ICC and the SBAIJ format reorderings are not available,
    since only the upper triangular part of the matrix is stored. You can use the
    SeqAIJ format in this case to get reorderings.

@*/
PetscErrorCode  PCFactorSetMatOrderingType(PC pc,const MatOrderingType ordering)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCFactorSetMatOrderingType_C",(PC,const MatOrderingType),(pc,ordering));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetColumnPivot"
/*@
    PCFactorSetColumnPivot - Determines when column pivoting is done during matrix factorization. 
      For PETSc dense matrices column pivoting is always done, for PETSc sparse matrices
      it is never done. For the MATLAB and SuperLU factorization this is used.

    Logically Collective on PC

    Input Parameters:
+   pc - the preconditioner context
-   dtcol - 0.0 implies no pivoting, 1.0 complete pivoting (slower, requires more memory but more stable)

    Options Database Key:
.   -pc_factor_pivoting <dtcol>

    Level: intermediate

.seealso: PCILUSetMatOrdering(), PCFactorSetPivotInBlocks()
@*/
PetscErrorCode  PCFactorSetColumnPivot(PC pc,PetscReal dtcol)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveReal(pc,dtcol,2);
  ierr = PetscTryMethod(pc,"PCFactorSetColumnPivot_C",(PC,PetscReal),(pc,dtcol));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetPivotInBlocks"
/*@
    PCFactorSetPivotInBlocks - Determines if pivoting is done while factoring each block
      with BAIJ or SBAIJ matrices

    Logically Collective on PC

    Input Parameters:
+   pc - the preconditioner context
-   pivot - PETSC_TRUE or PETSC_FALSE

    Options Database Key:
.   -pc_factor_pivot_in_blocks <true,false>

    Level: intermediate

.seealso: PCILUSetMatOrdering(), PCFactorSetColumnPivot()
@*/
PetscErrorCode  PCFactorSetPivotInBlocks(PC pc,PetscBool  pivot)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveBool(pc,pivot,2);
  ierr = PetscTryMethod(pc,"PCFactorSetPivotInBlocks_C",(PC,PetscBool),(pc,pivot));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetReuseFill"
/*@
   PCFactorSetReuseFill - When matrices with same different nonzero structure are factored,
   this causes later ones to use the fill ratio computed in the initial factorization.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  flag - PETSC_TRUE to reuse else PETSC_FALSE

   Options Database Key:
.  -pc_factor_reuse_fill - Activates PCFactorSetReuseFill()

   Level: intermediate

.keywords: PC, levels, reordering, factorization, incomplete, Cholesky

.seealso: PCFactorSetReuseOrdering()
@*/
PetscErrorCode  PCFactorSetReuseFill(PC pc,PetscBool  flag)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,2);
  PetscValidLogicalCollectiveBool(pc,flag,2);
  ierr = PetscTryMethod(pc,"PCFactorSetReuseFill_C",(PC,PetscBool),(pc,flag));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
