#define PETSCKSP_DLL

/*
   Defines a ILU factorization preconditioner for any Mat implementation
*/
#include "src/ksp/pc/pcimpl.h"                 /*I "petscpc.h"  I*/
#include "src/ksp/pc/impls/factor/ilu/ilu.h"
#include "src/mat/matimpl.h"

/* ------------------------------------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetZeroPivot_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetZeroPivot_ILU(PC pc,PetscReal z)
{
  PC_ILU *ilu;

  PetscFunctionBegin;
  ilu                 = (PC_ILU*)pc->data;
  ilu->info.zeropivot = z;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetShiftNonzero_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetShiftNonzero_ILU(PC pc,PetscReal shift)
{
  PC_ILU *dir;

  PetscFunctionBegin;
  dir = (PC_ILU*)pc->data;
  if (shift == (PetscReal) PETSC_DECIDE) {
    dir->info.shiftnz = 1.e-12;
  } else {
    dir->info.shiftnz = shift;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetShiftPd_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetShiftPd_ILU(PC pc,PetscTruth shift)
{
  PC_ILU *dir;
 
  PetscFunctionBegin;
  dir = (PC_ILU*)pc->data;
  dir->info.shiftpd = shift;
  if (shift) dir->info.shift_fraction = 0.0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCILUReorderForNonzeroDiagonal_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCILUReorderForNonzeroDiagonal_ILU(PC pc,PetscReal z)
{
  PC_ILU *ilu = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  ilu->nonzerosalongdiagonal = PETSC_TRUE;                 
  if (z == PETSC_DECIDE) {
    ilu->nonzerosalongdiagonaltol = 1.e-10;
  } else {
    ilu->nonzerosalongdiagonaltol = z;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_ILU_Internal"
PetscErrorCode PCDestroy_ILU_Internal(PC pc)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ilu->inplace && ilu->fact) {ierr = MatDestroy(ilu->fact);CHKERRQ(ierr);}
  if (ilu->row && ilu->col && ilu->row != ilu->col) {ierr = ISDestroy(ilu->row);CHKERRQ(ierr);}
  if (ilu->col) {ierr = ISDestroy(ilu->col);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCILUSetUseDropTolerance_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCILUSetUseDropTolerance_ILU(PC pc,PetscReal dt,PetscReal dtcol,PetscInt dtcount)
{
  PC_ILU         *ilu;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ilu = (PC_ILU*)pc->data;
  if (pc->setupcalled && (!ilu->usedt || ilu->info.dt != dt || ilu->info.dtcol != dtcol || ilu->info.dtcount != dtcount)) {
    pc->setupcalled   = 0;
    ierr = PCDestroy_ILU_Internal(pc);CHKERRQ(ierr);
  }
  ilu->usedt        = PETSC_TRUE;
  ilu->info.dt      = dt;
  ilu->info.dtcol   = dtcol;
  ilu->info.dtcount = dtcount;
  ilu->info.fill    = PETSC_DEFAULT;
  PetscFunctionReturn(0);
}  
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCILUSetFill_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCILUSetFill_ILU(PC pc,PetscReal fill)
{
  PC_ILU *dir;

  PetscFunctionBegin;
  dir            = (PC_ILU*)pc->data;
  dir->info.fill = fill;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCILUSetMatOrdering_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCILUSetMatOrdering_ILU(PC pc,MatOrderingType ordering)
{
  PC_ILU         *dir = (PC_ILU*)pc->data;
  PetscErrorCode ierr;
  PetscTruth     flg;
 
  PetscFunctionBegin;
  if (!pc->setupcalled) {
     ierr = PetscStrfree(dir->ordering);CHKERRQ(ierr);
     ierr = PetscStrallocpy(ordering,&dir->ordering);CHKERRQ(ierr);
  } else {
    ierr = PetscStrcmp(dir->ordering,ordering,&flg);CHKERRQ(ierr);
    if (!flg) {
      pc->setupcalled = 0;
      ierr = PetscStrfree(dir->ordering);CHKERRQ(ierr);
      ierr = PetscStrallocpy(ordering,&dir->ordering);CHKERRQ(ierr);
      /* free the data structures, then create them again */
      ierr = PCDestroy_ILU_Internal(pc);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCILUSetReuseOrdering_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCILUSetReuseOrdering_ILU(PC pc,PetscTruth flag)
{
  PC_ILU *ilu;

  PetscFunctionBegin;
  ilu                = (PC_ILU*)pc->data;
  ilu->reuseordering = flag;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCILUDTSetReuseFill_ILUDT"
PetscErrorCode PETSCKSP_DLLEXPORT PCILUDTSetReuseFill_ILUDT(PC pc,PetscTruth flag)
{
  PC_ILU *ilu;

  PetscFunctionBegin;
  ilu = (PC_ILU*)pc->data;
  ilu->reusefill = flag;
  if (flag) SETERRQ(PETSC_ERR_SUP,"Not yet supported");
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCILUSetLevels_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCILUSetLevels_ILU(PC pc,PetscInt levels)
{
  PC_ILU         *ilu;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ilu = (PC_ILU*)pc->data;

  if (!pc->setupcalled) {
    ilu->info.levels = levels;
  } else if (ilu->usedt || ilu->info.levels != levels) {
    ilu->info.levels = levels;
    pc->setupcalled  = 0;
    ilu->usedt       = PETSC_FALSE;
    ierr = PCDestroy_ILU_Internal(pc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCILUSetUseInPlace_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCILUSetUseInPlace_ILU(PC pc)
{
  PC_ILU *dir;

  PetscFunctionBegin;
  dir          = (PC_ILU*)pc->data;
  dir->inplace = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCILUSetAllowDiagonalFill"
PetscErrorCode PETSCKSP_DLLEXPORT PCILUSetAllowDiagonalFill_ILU(PC pc)
{
  PC_ILU *dir;

  PetscFunctionBegin;
  dir                 = (PC_ILU*)pc->data;
  dir->info.diagonal_fill = 1;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCILUSetUseDropTolerance"
/*@
   PCILUSetUseDropTolerance - The preconditioner will use an ILU 
   based on a drop tolerance.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  dt - the drop tolerance, try from 1.e-10 to .1
.  dtcol - tolerance for column pivot, good values [0.1 to 0.01]
-  maxrowcount - the max number of nonzeros allowed in a row, best value
                 depends on the number of nonzeros in row of original matrix

   Options Database Key:
.  -pc_ilu_use_drop_tolerance <dt,dtcol,maxrowcount> - Sets drop tolerance

   Level: intermediate

    Notes:
      This uses the iludt() code of Saad's SPARSKIT package

.keywords: PC, levels, reordering, factorization, incomplete, ILU
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCILUSetUseDropTolerance(PC pc,PetscReal dt,PetscReal dtcol,PetscInt maxrowcount)
{
  PetscErrorCode ierr,(*f)(PC,PetscReal,PetscReal,PetscInt);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCILUSetUseDropTolerance_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,dt,dtcol,maxrowcount);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}  

#undef __FUNCT__  
#define __FUNCT__ "PCILUSetFill"
/*@
   PCILUSetFill - Indicate the amount of fill you expect in the factored matrix,
   where fill = number nonzeros in factor/number nonzeros in original matrix.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  fill - amount of expected fill

   Options Database Key:
$  -pc_ilu_fill <fill>

   Note:
   For sparse matrix factorizations it is difficult to predict how much 
   fill to expect. By running with the option -log_info PETSc will print the 
   actual amount of fill used; allowing you to set the value accurately for
   future runs. But default PETSc uses a value of 1.0

   Level: intermediate

.keywords: PC, set, factorization, direct, fill

.seealso: PCLUSetFill()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCILUSetFill(PC pc,PetscReal fill)
{
  PetscErrorCode ierr,(*f)(PC,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (fill < 1.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Fill factor cannot be less than 1.0");
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCILUSetFill_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,fill);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCILUSetMatOrdering"
/*@C
    PCILUSetMatOrdering - Sets the ordering routine (to reduce fill) to 
    be used in the ILU factorization.

    Collective on PC

    Input Parameters:
+   pc - the preconditioner context
-   ordering - the matrix ordering name, for example, MATORDERING_ND or MATORDERING_RCM

    Options Database Key:
.   -pc_ilu_mat_ordering_type <nd,rcm,...> - Sets ordering routine

    Level: intermediate

    Notes: natural ordering is used by default

.seealso: PCLUSetMatOrdering()

.keywords: PC, ILU, set, matrix, reordering

@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCILUSetMatOrdering(PC pc,MatOrderingType ordering)
{
  PetscErrorCode ierr,(*f)(PC,MatOrderingType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCILUSetMatOrdering_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,ordering);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCILUSetReuseOrdering"
/*@
   PCILUSetReuseOrdering - When similar matrices are factored, this
   causes the ordering computed in the first factor to be used for all
   following factors; applies to both fill and drop tolerance ILUs.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  flag - PETSC_TRUE to reuse else PETSC_FALSE

   Options Database Key:
.  -pc_ilu_reuse_ordering - Activate PCILUSetReuseOrdering()

   Level: intermediate

.keywords: PC, levels, reordering, factorization, incomplete, ILU

.seealso: PCILUDTSetReuseFill(), PCLUSetReuseOrdering(), PCLUSetReuseFill()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCILUSetReuseOrdering(PC pc,PetscTruth flag)
{
  PetscErrorCode ierr,(*f)(PC,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCILUSetReuseOrdering_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,flag);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCILUDTSetReuseFill"
/*@
   PCILUDTSetReuseFill - When matrices with same nonzero structure are ILUDT factored,
   this causes later ones to use the fill computed in the initial factorization.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  flag - PETSC_TRUE to reuse else PETSC_FALSE

   Options Database Key:
.  -pc_iludt_reuse_fill - Activates PCILUDTSetReuseFill()

   Level: intermediate

.keywords: PC, levels, reordering, factorization, incomplete, ILU

.seealso: PCILUSetReuseOrdering(), PCLUSetReuseOrdering(), PCLUSetReuseFill()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCILUDTSetReuseFill(PC pc,PetscTruth flag)
{
  PetscErrorCode ierr,(*f)(PC,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCILUDTSetReuseFill_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,flag);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCILUSetLevels"
/*@
   PCILUSetLevels - Sets the number of levels of fill to use.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  levels - number of levels of fill

   Options Database Key:
.  -pc_ilu_levels <levels> - Sets fill level

   Level: intermediate

.keywords: PC, levels, fill, factorization, incomplete, ILU
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCILUSetLevels(PC pc,PetscInt levels)
{
  PetscErrorCode ierr,(*f)(PC,PetscInt);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (levels < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"negative levels");
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCILUSetLevels_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,levels);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCILUSetAllowDiagonalFill"
/*@
   PCILUSetAllowDiagonalFill - Causes all diagonal matrix entries to be 
   treated as level 0 fill even if there is no non-zero location.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context

   Options Database Key:
.  -pc_ilu_diagonal_fill

   Notes:
   Does not apply with 0 fill.

   Level: intermediate

.keywords: PC, levels, fill, factorization, incomplete, ILU
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCILUSetAllowDiagonalFill(PC pc)
{
  PetscErrorCode ierr,(*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCILUSetAllowDiagonalFill_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCILUSetUseInPlace"
/*@
   PCILUSetUseInPlace - Tells the system to do an in-place incomplete factorization.
   Collective on PC

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
.  -pc_ilu_in_place - Activates in-place factorization

   Notes:
   PCILUSetUseInPlace() is intended for use with matrix-free variants of
   Krylov methods, or when a different matrices are employed for the linear
   system and preconditioner, or with ASM preconditioning.  Do NOT use 
   this option if the linear system
   matrix also serves as the preconditioning matrix, since the factored
   matrix would then overwrite the original matrix. 

   Only works well with ILU(0).

   Level: intermediate

.keywords: PC, set, factorization, inplace, in-place, ILU

.seealso:  PCLUSetUseInPlace()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCILUSetUseInPlace(PC pc)
{
  PetscErrorCode ierr,(*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCILUSetUseInPlace_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCILUReorderForNonzeroDiagonal"
/*@
   PCILUReorderForNonzeroDiagonal - reorders rows/columns of matrix to remove zeros from diagonal

   Collective on PC
   
   Input Parameters:
+  pc - the preconditioner context
-  tol - diagonal entries smaller than this in absolute value are considered zero

   Options Database Key:
.  -pc_lu_nonzeros_along_diagonal

   Level: intermediate

.keywords: PC, set, factorization, direct, fill

.seealso: PCILUSetFill(), MatReorderForNonzeroDiagonal()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCILUReorderForNonzeroDiagonal(PC pc,PetscReal rtol)
{
  PetscErrorCode ierr,(*f)(PC,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCILUReorderForNonzeroDiagonal_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,rtol);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCILUSetPivotInBlocks"
/*@
    PCILUSetPivotInBlocks - Determines if pivoting is done while factoring each block
      with BAIJ or SBAIJ matrices

    Collective on PC

    Input Parameters:
+   pc - the preconditioner context
-   pivot - PETSC_TRUE or PETSC_FALSE

    Options Database Key:
.   -pc_ilu_pivot_in_blocks <true,false>

    Level: intermediate

.seealso: PCIILUSetMatOrdering(), PCILUSetPivoting()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCILUSetPivotInBlocks(PC pc,PetscTruth pivot)
{
  PetscErrorCode ierr,(*f)(PC,PetscTruth);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCILUSetPivotInBlocks_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,pivot);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCILUSetPivotInBlocks_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCILUSetPivotInBlocks_ILU(PC pc,PetscTruth pivot)
{
  PC_ILU *dir = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  dir->info.pivotinblocks = pivot ? 1.0 : 0.0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_ILU"
static PetscErrorCode PCSetFromOptions_ILU(PC pc)
{
  PetscErrorCode ierr;
  PetscInt       dtmax = 3,itmp;
  PetscTruth     flg,set;
  PetscReal      dt[3];
  char           tname[256];
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscFList     ordlist;
  PetscReal      tol;

  PetscFunctionBegin;
  ierr = MatOrderingRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHead("ILU Options");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-pc_ilu_levels","levels of fill","PCILUSetLevels",(PetscInt)ilu->info.levels,&itmp,&flg);CHKERRQ(ierr);
    if (flg) ilu->info.levels = itmp;
    ierr = PetscOptionsName("-pc_ilu_in_place","do factorization in place","PCILUSetUseInPlace",&ilu->inplace);CHKERRQ(ierr);
    ierr = PetscOptionsName("-pc_ilu_diagonal_fill","Allow fill into empty diagonal entry","PCILUSetAllowDiagonalFill",&flg);CHKERRQ(ierr);
    ilu->info.diagonal_fill = (double) flg;
    ierr = PetscOptionsName("-pc_iludt_reuse_fill","Reuse fill from previous ILUdt","PCILUDTSetReuseFill",&ilu->reusefill);CHKERRQ(ierr);
    ierr = PetscOptionsName("-pc_ilu_reuse_ordering","Reuse previous reordering","PCILUSetReuseOrdering",&ilu->reuseordering);CHKERRQ(ierr);

    ierr = PetscOptionsName("-pc_factor_shift_nonzero","Shift added to diagonal","PCFactorSetShiftNonzero",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCFactorSetShiftNonzero(pc,(PetscReal)PETSC_DECIDE);CHKERRQ(ierr);
    }
    ierr = PetscOptionsReal("-pc_factor_shift_nonzero","Shift added to diagonal","PCFactorSetShiftNonzero",ilu->info.shiftnz,&ilu->info.shiftnz,0);CHKERRQ(ierr);
    
    ierr = PetscOptionsName("-pc_factor_shift_positive_definite","Manteuffel shift applied to diagonal","PCFactorSetShiftPd",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscOptionsInt("-pc_factor_shift_positive_definite","Manteuffel shift applied to diagonal","PCFactorSetShiftPd",(PetscInt)ilu->info.shiftpd,&itmp,&flg); CHKERRQ(ierr);
      if (flg && !itmp) {
	ierr = PCFactorSetShiftPd(pc,PETSC_FALSE);CHKERRQ(ierr);
      } else {
	ierr = PCFactorSetShiftPd(pc,PETSC_TRUE);CHKERRQ(ierr); 
      }
    }
    ierr = PetscOptionsReal("-pc_factor_zeropivot","Pivot is considered zero if less than","PCFactorSetZeroPivot",ilu->info.zeropivot,&ilu->info.zeropivot,0);CHKERRQ(ierr);

    dt[0] = ilu->info.dt;
    dt[1] = ilu->info.dtcol;
    dt[2] = ilu->info.dtcount;
    ierr = PetscOptionsRealArray("-pc_ilu_use_drop_tolerance","<dt,dtcol,maxrowcount>","PCILUSetUseDropTolerance",dt,&dtmax,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCILUSetUseDropTolerance(pc,dt[0],dt[1],(PetscInt)dt[2]);CHKERRQ(ierr);
    }
    ierr = PetscOptionsReal("-pc_ilu_fill","Expected fill in factorization","PCILUSetFill",ilu->info.fill,&ilu->info.fill,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsName("-pc_ilu_nonzeros_along_diagonal","Reorder to remove zeros from diagonal","PCILUReorderForNonzeroDiagonal",&flg);CHKERRQ(ierr);
    if (flg) {
      tol = PETSC_DECIDE;
      ierr = PetscOptionsReal("-pc_ilu_nonzeros_along_diagonal","Reorder to remove zeros from diagonal","PCILUReorderForNonzeroDiagonal",ilu->nonzerosalongdiagonaltol,&tol,0);CHKERRQ(ierr);
      ierr = PCILUReorderForNonzeroDiagonal(pc,tol);CHKERRQ(ierr);
    }

    ierr = MatGetOrderingList(&ordlist);CHKERRQ(ierr);
    ierr = PetscOptionsList("-pc_ilu_mat_ordering_type","Reorder to reduce nonzeros in ILU","PCILUSetMatOrdering",ordlist,ilu->ordering,tname,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCILUSetMatOrdering(pc,tname);CHKERRQ(ierr);
    }
    flg = ilu->info.pivotinblocks ? PETSC_TRUE : PETSC_FALSE;
    ierr = PetscOptionsLogical("-pc_ilu_pivot_in_blocks","Pivot inside matrix blocks for BAIJ and SBAIJ","PCILUSetPivotInBlocks",flg,&flg,&set);CHKERRQ(ierr);
    if (set) {
      ierr = PCILUSetPivotInBlocks(pc,flg);CHKERRQ(ierr);
    }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCView_ILU"
static PetscErrorCode PCView_ILU(PC pc,PetscViewer viewer)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscErrorCode ierr;
  PetscTruth     isstring,iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (ilu->usedt) {
        ierr = PetscViewerASCIIPrintf(viewer,"  ILU: drop tolerance %g\n",ilu->info.dt);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  ILU: max nonzeros per row %D\n",(PetscInt)ilu->info.dtcount);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  ILU: column permutation tolerance %g\n",ilu->info.dtcol);CHKERRQ(ierr);
    } else if (ilu->info.levels == 1) {
        ierr = PetscViewerASCIIPrintf(viewer,"  ILU: %D level of fill\n",(PetscInt)ilu->info.levels);CHKERRQ(ierr);
    } else {
        ierr = PetscViewerASCIIPrintf(viewer,"  ILU: %D levels of fill\n",(PetscInt)ilu->info.levels);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  ILU: max fill ratio allocated %g\n",ilu->info.fill);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  ILU: tolerance for zero pivot %g\n",ilu->info.zeropivot);CHKERRQ(ierr);
    if (ilu->info.shiftpd) {ierr = PetscViewerASCIIPrintf(viewer,"  ILU: using Manteuffel shift\n");CHKERRQ(ierr);}
    if (ilu->inplace) {ierr = PetscViewerASCIIPrintf(viewer,"       in-place factorization\n");CHKERRQ(ierr);}
    else              {ierr = PetscViewerASCIIPrintf(viewer,"       out-of-place factorization\n");CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer,"       matrix ordering: %s\n",ilu->ordering);CHKERRQ(ierr);
    if (ilu->reusefill)     {ierr = PetscViewerASCIIPrintf(viewer,"       Reusing fill from past factorization\n");CHKERRQ(ierr);}
    if (ilu->reuseordering) {ierr = PetscViewerASCIIPrintf(viewer,"       Reusing reordering from past factorization\n");CHKERRQ(ierr);}
    if (ilu->fact) {
      ierr = PetscViewerASCIIPrintf(viewer,"       Factored matrix follows\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
      ierr = MatView(ilu->fact,viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer," lvls=%D,order=%s",(PetscInt)ilu->info.levels,ilu->ordering);CHKERRQ(ierr);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for PCILU",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_ILU"
static PetscErrorCode PCSetUp_ILU(PC pc)
{
  PetscErrorCode ierr;
  PC_ILU         *ilu = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  if (ilu->inplace) {
    if (!pc->setupcalled) {

      /* In-place factorization only makes sense with the natural ordering,
         so we only need to get the ordering once, even if nonzero structure changes */
      ierr = MatGetOrdering(pc->pmat,ilu->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
      if (ilu->row) {ierr = PetscLogObjectParent(pc,ilu->row);CHKERRQ(ierr);}
      if (ilu->col) {ierr = PetscLogObjectParent(pc,ilu->col);CHKERRQ(ierr);}
    }

    /* In place ILU only makes sense with fill factor of 1.0 because 
       cannot have levels of fill */
    ilu->info.fill          = 1.0;
    ilu->info.diagonal_fill = 0;
    ierr = MatILUFactor(pc->pmat,ilu->row,ilu->col,&ilu->info);CHKERRQ(ierr);
    ilu->fact = pc->pmat;
  } else if (ilu->usedt) {
    if (!pc->setupcalled) {
      ierr = MatGetOrdering(pc->pmat,ilu->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
      if (ilu->row) {ierr = PetscLogObjectParent(pc,ilu->row);CHKERRQ(ierr);}
      if (ilu->col) {ierr = PetscLogObjectParent(pc,ilu->col);CHKERRQ(ierr);}
      ierr = MatILUDTFactor(pc->pmat,ilu->row,ilu->col,&ilu->info,&ilu->fact);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(pc,ilu->fact);CHKERRQ(ierr);
    } else if (pc->flag != SAME_NONZERO_PATTERN) { 
      ierr = MatDestroy(ilu->fact);CHKERRQ(ierr);
      if (!ilu->reuseordering) {
        if (ilu->row) {ierr = ISDestroy(ilu->row);CHKERRQ(ierr);}
        if (ilu->col) {ierr = ISDestroy(ilu->col);CHKERRQ(ierr);}
        ierr = MatGetOrdering(pc->pmat,ilu->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
        if (ilu->row) {ierr = PetscLogObjectParent(pc,ilu->row);CHKERRQ(ierr);}
        if (ilu->col) {ierr = PetscLogObjectParent(pc,ilu->col);CHKERRQ(ierr);}
      }
      ierr = MatILUDTFactor(pc->pmat,ilu->row,ilu->col,&ilu->info,&ilu->fact);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(pc,ilu->fact);CHKERRQ(ierr);
    } else if (!ilu->reusefill) { 
      ierr = MatDestroy(ilu->fact);CHKERRQ(ierr);
      ierr = MatILUDTFactor(pc->pmat,ilu->row,ilu->col,&ilu->info,&ilu->fact);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(pc,ilu->fact);CHKERRQ(ierr);
    } else {
      ierr = MatLUFactorNumeric(pc->pmat,&ilu->info,&ilu->fact);CHKERRQ(ierr);
    }
   } else {
    if (!pc->setupcalled) {
      /* first time in so compute reordering and symbolic factorization */
      ierr = MatGetOrdering(pc->pmat,ilu->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
      if (ilu->row) {ierr = PetscLogObjectParent(pc,ilu->row);CHKERRQ(ierr);}
      if (ilu->col) {ierr = PetscLogObjectParent(pc,ilu->col);CHKERRQ(ierr);}
      /*  Remove zeros along diagonal?     */
      if (ilu->nonzerosalongdiagonal) {
        ierr = MatReorderForNonzeroDiagonal(pc->pmat,ilu->nonzerosalongdiagonaltol,ilu->row,ilu->col);CHKERRQ(ierr);
      }
      ierr = MatILUFactorSymbolic(pc->pmat,ilu->row,ilu->col,&ilu->info,&ilu->fact);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(pc,ilu->fact);CHKERRQ(ierr);
    } else if (pc->flag != SAME_NONZERO_PATTERN) { 
      if (!ilu->reuseordering) {
        /* compute a new ordering for the ILU */
        ierr = ISDestroy(ilu->row);CHKERRQ(ierr);
        ierr = ISDestroy(ilu->col);CHKERRQ(ierr);
        ierr = MatGetOrdering(pc->pmat,ilu->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
        if (ilu->row) {ierr = PetscLogObjectParent(pc,ilu->row);CHKERRQ(ierr);}
        if (ilu->col) {ierr = PetscLogObjectParent(pc,ilu->col);CHKERRQ(ierr);}
        /*  Remove zeros along diagonal?     */
        if (ilu->nonzerosalongdiagonal) {
          ierr = MatReorderForNonzeroDiagonal(pc->pmat,ilu->nonzerosalongdiagonaltol,ilu->row,ilu->col);CHKERRQ(ierr);
        }
      }
      ierr = MatDestroy(ilu->fact);CHKERRQ(ierr);
      ierr = MatILUFactorSymbolic(pc->pmat,ilu->row,ilu->col,&ilu->info,&ilu->fact);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(pc,ilu->fact);CHKERRQ(ierr);
    }
    ierr = MatLUFactorNumeric(pc->pmat,&ilu->info,&ilu->fact);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_ILU"
static PetscErrorCode PCDestroy_ILU(PC pc)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCDestroy_ILU_Internal(pc);CHKERRQ(ierr);
  ierr = PetscStrfree(ilu->ordering);CHKERRQ(ierr);
  ierr = PetscFree(ilu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_ILU"
static PetscErrorCode PCApply_ILU(PC pc,Vec x,Vec y)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolve(ilu->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyTranspose_ILU"
static PetscErrorCode PCApplyTranspose_ILU(PC pc,Vec x,Vec y)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolveTranspose(ilu->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCGetFactoredMatrix_ILU"
static PetscErrorCode PCGetFactoredMatrix_ILU(PC pc,Mat *mat)
{
  PC_ILU *ilu = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  if (!ilu->fact) SETERRQ(PETSC_ERR_ORDER,"Matrix not yet factored; call after KSPSetUp() or PCSetUp()");
  *mat = ilu->fact;
  PetscFunctionReturn(0);
}

/*MC
     PCILU - Incomplete factorization preconditioners.

   Options Database Keys:
+  -pc_ilu_levels <k> - number of levels of fill for ILU(k)
.  -pc_ilu_in_place - only for ILU(0) with natural ordering, reuses the space of the matrix for
                      its factorization (overwrites original matrix)
.  -pc_ilu_diagonal_fill - fill in a zero diagonal even if levels of fill indicate it wouldn't be fill
.  -pc_ilu_reuse_ordering - reuse ordering of factorized matrix from previous factorization
.  -pc_ilu_use_drop_tolerance <dt,dtcol,maxrowcount> - use Saad's drop tolerance ILUdt
.  -pc_ilu_fill <nfill> - expected amount of fill in factored matrix compared to original matrix, nfill > 1
.  -pc_ilu_nonzeros_along_diagonal - reorder the matrix before factorization to remove zeros from the diagonal,
                                   this decreases the chance of getting a zero pivot
.  -pc_ilu_mat_ordering_type <natural,nd,1wd,rcm,qmd> - set the row/column ordering of the factored matrix
-  -pc_ilu_pivot_in_blocks - for block ILU(k) factorization, i.e. with BAIJ matrices with block size larger
                             than 1 the diagonal blocks are factored with partial pivoting (this increases the 
                             stability of the ILU factorization

   Level: beginner

  Concepts: incomplete factorization

   Notes: Only implemented for some matrix formats. Not implemented in parallel

          For BAIJ matrices this implements a point block ILU

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCSOR, MatOrderingType,
           PCFactorSetZeroPivot(), PCFactorSetShiftNonzero(), PCFactorSetShiftPd(), PCILUSetUseDropTolerance(),
           PCILUSetFill(), PCILUSetMatOrdering(), PCILUSetReuseOrdering(), PCILUDTSetReuseFill(),
           PCILUSetLevels(), PCILUSetUseInPlace(), PCILUSetAllowDiagonalFill(), PCILUSetPivotInBlocks(),

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_ILU(PC pc)
{
  PetscErrorCode ierr;
  PC_ILU         *ilu;

  PetscFunctionBegin;
  ierr = PetscNew(PC_ILU,&ilu);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(pc,sizeof(PC_ILU));CHKERRQ(ierr);

  ilu->fact                    = 0;
  ierr = MatFactorInfoInitialize(&ilu->info);CHKERRQ(ierr);
  ilu->info.levels             = 0;
  ilu->info.fill               = 1.0; 
  ilu->col                     = 0;
  ilu->row                     = 0;
  ilu->inplace                 = PETSC_FALSE;
  ierr = PetscStrallocpy(MATORDERING_NATURAL,&ilu->ordering);CHKERRQ(ierr);
  ilu->reuseordering           = PETSC_FALSE;
  ilu->usedt                   = PETSC_FALSE;
  ilu->info.dt                 = PETSC_DEFAULT;
  ilu->info.dtcount            = PETSC_DEFAULT;
  ilu->info.dtcol              = PETSC_DEFAULT;
  ilu->info.shiftnz            = 0.0;
  ilu->info.shiftpd            = PETSC_FALSE;
  ilu->info.shift_fraction     = 0.0;
  ilu->info.zeropivot          = 1.e-12;
  ilu->info.pivotinblocks      = 1.0;
  ilu->reusefill               = PETSC_FALSE;
  ilu->info.diagonal_fill      = 0;
  pc->data                     = (void*)ilu;

  pc->ops->destroy             = PCDestroy_ILU;
  pc->ops->apply               = PCApply_ILU;
  pc->ops->applytranspose      = PCApplyTranspose_ILU;
  pc->ops->setup               = PCSetUp_ILU;
  pc->ops->setfromoptions      = PCSetFromOptions_ILU;
  pc->ops->getfactoredmatrix   = PCGetFactoredMatrix_ILU;
  pc->ops->view                = PCView_ILU;
  pc->ops->applyrichardson     = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetZeroPivot_C","PCFactorSetZeroPivot_ILU",
                    PCFactorSetZeroPivot_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetShiftNonzero_C","PCFactorSetShiftNonzero_ILU",
                    PCFactorSetShiftNonzero_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetShiftPd_C","PCFactorSetShiftPd_ILU",
                    PCFactorSetShiftPd_ILU);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCILUSetUseDropTolerance_C","PCILUSetUseDropTolerance_ILU",
                    PCILUSetUseDropTolerance_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCILUSetFill_C","PCILUSetFill_ILU",
                    PCILUSetFill_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCILUSetMatOrdering_C","PCILUSetMatOrdering_ILU",
                    PCILUSetMatOrdering_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCILUSetReuseOrdering_C","PCILUSetReuseOrdering_ILU",
                    PCILUSetReuseOrdering_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCILUDTSetReuseFill_C","PCILUDTSetReuseFill_ILUDT",
                    PCILUDTSetReuseFill_ILUDT);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCILUSetLevels_C","PCILUSetLevels_ILU",
                    PCILUSetLevels_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCILUSetUseInPlace_C","PCILUSetUseInPlace_ILU",
                    PCILUSetUseInPlace_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCILUSetAllowDiagonalFill_C","PCILUSetAllowDiagonalFill_ILU",
                    PCILUSetAllowDiagonalFill_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCILUSetPivotInBlocks_C","PCILUSetPivotInBlocks_ILU",
                    PCILUSetPivotInBlocks_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCILUReorderForNonzeroDiagonal_C","PCILUReorderForNonzeroDiagonal_ILU",
                    PCILUReorderForNonzeroDiagonal_ILU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
