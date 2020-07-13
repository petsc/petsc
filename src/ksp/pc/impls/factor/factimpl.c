
#include <../src/ksp/pc/impls/factor/factor.h>     /*I "petscpc.h"  I*/

/* ------------------------------------------------------------------------------------------*/


PetscErrorCode PCFactorSetUpMatSolverType_Factor(PC pc)
{
  PC_Factor      *icc = (PC_Factor*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!pc->pmat) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"You can only call this routine after the matrix object has been provided to the solver, for example with KSPSetOperators() or SNESSetJacobian()");
  if (!pc->setupcalled && !((PC_Factor*)icc)->fact) {
    ierr = MatGetFactor(pc->pmat,((PC_Factor*)icc)->solvertype,((PC_Factor*)icc)->factortype,&((PC_Factor*)icc)->fact);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  PCFactorSetZeroPivot_Factor(PC pc,PetscReal z)
{
  PC_Factor *ilu = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  ilu->info.zeropivot = z;
  PetscFunctionReturn(0);
}

PetscErrorCode  PCFactorSetShiftType_Factor(PC pc,MatFactorShiftType shifttype)
{
  PC_Factor *dir = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  if (shifttype == (MatFactorShiftType)PETSC_DECIDE) dir->info.shifttype = (PetscReal) MAT_SHIFT_NONE;
  else {
    dir->info.shifttype = (PetscReal) shifttype;
    if ((shifttype == MAT_SHIFT_NONZERO || shifttype ==  MAT_SHIFT_INBLOCKS) && dir->info.shiftamount == 0.0) {
      dir->info.shiftamount = 100.0*PETSC_MACHINE_EPSILON; /* set default amount if user has not called PCFactorSetShiftAmount() yet */
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  PCFactorSetShiftAmount_Factor(PC pc,PetscReal shiftamount)
{
  PC_Factor *dir = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  if (shiftamount == (PetscReal) PETSC_DECIDE) dir->info.shiftamount = 100.0*PETSC_MACHINE_EPSILON;
  else dir->info.shiftamount = shiftamount;
  PetscFunctionReturn(0);
}

PetscErrorCode  PCFactorSetDropTolerance_Factor(PC pc,PetscReal dt,PetscReal dtcol,PetscInt dtcount)
{
  PC_Factor *ilu = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  if (pc->setupcalled && (!ilu->info.usedt || ((PC_Factor*)ilu)->info.dt != dt || ((PC_Factor*)ilu)->info.dtcol != dtcol || ((PC_Factor*)ilu)->info.dtcount != dtcount)) {
    SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Cannot change tolerance after use");
  }
  ilu->info.usedt   = PETSC_TRUE;
  ilu->info.dt      = dt;
  ilu->info.dtcol   = dtcol;
  ilu->info.dtcount = dtcount;
  ilu->info.fill    = PETSC_DEFAULT;
  PetscFunctionReturn(0);
}

PetscErrorCode  PCFactorSetFill_Factor(PC pc,PetscReal fill)
{
  PC_Factor *dir = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  dir->info.fill = fill;
  PetscFunctionReturn(0);
}

PetscErrorCode  PCFactorSetMatOrderingType_Factor(PC pc,MatOrderingType ordering)
{
  PC_Factor      *dir = (PC_Factor*)pc->data;
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    ierr = PetscFree(dir->ordering);CHKERRQ(ierr);
    ierr = PetscStrallocpy(ordering,(char**)&dir->ordering);CHKERRQ(ierr);
  } else {
    ierr = PetscStrcmp(dir->ordering,ordering,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Cannot change ordering after use");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  PCFactorGetLevels_Factor(PC pc,PetscInt *levels)
{
  PC_Factor      *ilu = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  *levels = ilu->info.levels;
  PetscFunctionReturn(0);
}

PetscErrorCode  PCFactorGetZeroPivot_Factor(PC pc,PetscReal *pivot)
{
  PC_Factor      *ilu = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  *pivot = ilu->info.zeropivot;
  PetscFunctionReturn(0);
}

PetscErrorCode  PCFactorGetShiftAmount_Factor(PC pc,PetscReal *shift)
{
  PC_Factor      *ilu = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  *shift = ilu->info.shiftamount;
  PetscFunctionReturn(0);
}

PetscErrorCode  PCFactorGetShiftType_Factor(PC pc,MatFactorShiftType *type)
{
  PC_Factor      *ilu = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  *type = (MatFactorShiftType) (int) ilu->info.shifttype;
  PetscFunctionReturn(0);
}

PetscErrorCode  PCFactorSetLevels_Factor(PC pc,PetscInt levels)
{
  PC_Factor      *ilu = (PC_Factor*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!pc->setupcalled) ilu->info.levels = levels;
  else if (ilu->info.levels != levels) {
    ierr             = (*pc->ops->reset)(pc);CHKERRQ(ierr); /* remove previous factored matrices */
    pc->setupcalled  = 0; /* force a complete rebuild of preconditioner factored matrices */
    ilu->info.levels = levels;
  } else if (ilu->info.usedt) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Cannot change levels after use with ILUdt");
  PetscFunctionReturn(0);
}

PetscErrorCode  PCFactorSetAllowDiagonalFill_Factor(PC pc,PetscBool flg)
{
  PC_Factor *dir = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  dir->info.diagonal_fill = (PetscReal) flg;
  PetscFunctionReturn(0);
}

PetscErrorCode  PCFactorGetAllowDiagonalFill_Factor(PC pc,PetscBool *flg)
{
  PC_Factor *dir = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  *flg = dir->info.diagonal_fill ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------------*/

PetscErrorCode  PCFactorSetPivotInBlocks_Factor(PC pc,PetscBool pivot)
{
  PC_Factor *dir = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  dir->info.pivotinblocks = pivot ? 1.0 : 0.0;
  PetscFunctionReturn(0);
}

PetscErrorCode  PCFactorGetMatrix_Factor(PC pc,Mat *mat)
{
  PC_Factor *ilu = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  if (!ilu->fact) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ORDER,"Matrix not yet factored; call after KSPSetUp() or PCSetUp()");
  *mat = ilu->fact;
  PetscFunctionReturn(0);
}

PetscErrorCode  PCFactorSetMatSolverType_Factor(PC pc,MatSolverType stype)
{
  PetscErrorCode ierr;
  PC_Factor      *lu = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  if (lu->fact) { 
    MatSolverType ltype;
    PetscBool     flg;
    ierr = MatFactorGetSolverType(lu->fact,&ltype);CHKERRQ(ierr);
    ierr = PetscStrcmp(stype,ltype,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Cannot change solver matrix package after PC has been setup or used");
  } 
 
  ierr = PetscFree(lu->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(stype,&lu->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  PCFactorGetMatSolverType_Factor(PC pc,MatSolverType *stype)
{
  PC_Factor *lu = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  *stype = lu->solvertype;
  PetscFunctionReturn(0);
}

PetscErrorCode  PCFactorSetColumnPivot_Factor(PC pc,PetscReal dtcol)
{
  PC_Factor *dir = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  if (dtcol < 0.0 || dtcol > 1.0) SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Column pivot tolerance is %g must be between 0 and 1",(double)dtcol);
  dir->info.dtcol = dtcol;
  PetscFunctionReturn(0);
}

PetscErrorCode  PCSetFromOptions_Factor(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_Factor         *factor = (PC_Factor*)pc->data;
  PetscErrorCode    ierr;
  PetscBool         flg,set;
  char              tname[256], solvertype[64];
  PetscFunctionList ordlist;
  PetscEnum         etmp;
  PetscBool         inplace;

  PetscFunctionBegin;
  ierr = PCFactorGetUseInPlace(pc,&inplace);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_factor_in_place","Form factored matrix in the same memory as the matrix","PCFactorSetUseInPlace",inplace,&flg,&set);CHKERRQ(ierr);
  if (set) {
    ierr = PCFactorSetUseInPlace(pc,flg);CHKERRQ(ierr);
  }
  ierr = PetscOptionsReal("-pc_factor_fill","Expected non-zeros in factored matrix","PCFactorSetFill",((PC_Factor*)factor)->info.fill,&((PC_Factor*)factor)->info.fill,NULL);CHKERRQ(ierr);

  ierr = PetscOptionsEnum("-pc_factor_shift_type","Type of shift to add to diagonal","PCFactorSetShiftType",MatFactorShiftTypes,(PetscEnum)(int)((PC_Factor*)factor)->info.shifttype,&etmp,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCFactorSetShiftType(pc,(MatFactorShiftType)etmp);CHKERRQ(ierr);
  }
  ierr = PetscOptionsReal("-pc_factor_shift_amount","Shift added to diagonal","PCFactorSetShiftAmount",((PC_Factor*)factor)->info.shiftamount,&((PC_Factor*)factor)->info.shiftamount,NULL);CHKERRQ(ierr);

  ierr = PetscOptionsReal("-pc_factor_zeropivot","Pivot is considered zero if less than","PCFactorSetZeroPivot",((PC_Factor*)factor)->info.zeropivot,&((PC_Factor*)factor)->info.zeropivot,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_factor_column_pivot","Column pivot tolerance (used only for some factorization)","PCFactorSetColumnPivot",((PC_Factor*)factor)->info.dtcol,&((PC_Factor*)factor)->info.dtcol,&flg);CHKERRQ(ierr);

  ierr = PetscOptionsBool("-pc_factor_pivot_in_blocks","Pivot inside matrix dense blocks for BAIJ and SBAIJ","PCFactorSetPivotInBlocks",((PC_Factor*)factor)->info.pivotinblocks ? PETSC_TRUE : PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
  if (set) {
    ierr = PCFactorSetPivotInBlocks(pc,flg);CHKERRQ(ierr);
  }

  ierr = PetscOptionsBool("-pc_factor_reuse_fill","Use fill from previous factorization","PCFactorSetReuseFill",PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
  if (set) {
    ierr = PCFactorSetReuseFill(pc,flg);CHKERRQ(ierr);
  }
  ierr = PetscOptionsBool("-pc_factor_reuse_ordering","Reuse ordering from previous factorization","PCFactorSetReuseOrdering",PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
  if (set) {
    ierr = PCFactorSetReuseOrdering(pc,flg);CHKERRQ(ierr);
  }

  ierr = MatGetOrderingList(&ordlist);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-pc_factor_mat_ordering_type","Reordering to reduce nonzeros in factored matrix","PCFactorSetMatOrderingType",ordlist,((PC_Factor*)factor)->ordering,tname,sizeof(tname),&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCFactorSetMatOrderingType(pc,tname);CHKERRQ(ierr);
  }

  /* maybe should have MatGetSolverTypes(Mat,&list) like the ordering list */
  ierr = PetscOptionsDeprecated("-pc_factor_mat_solver_package","-pc_factor_mat_solver_type","3.9",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-pc_factor_mat_solver_type","Specific direct solver to use","MatGetFactor",((PC_Factor*)factor)->solvertype,solvertype,sizeof(solvertype),&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCFactorSetMatSolverType(pc,solvertype);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCView_Factor(PC pc,PetscViewer viewer)
{
  PC_Factor       *factor = (PC_Factor*)pc->data;
  PetscErrorCode  ierr;
  PetscBool       isstring,iascii,flg;
  MatOrderingType ordering;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (factor->inplace) {
      ierr = PetscViewerASCIIPrintf(viewer,"  in-place factorization\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  out-of-place factorization\n");CHKERRQ(ierr);
    }

    if (factor->reusefill)     {ierr = PetscViewerASCIIPrintf(viewer,"  Reusing fill from past factorization\n");CHKERRQ(ierr);}
    if (factor->reuseordering) {ierr = PetscViewerASCIIPrintf(viewer,"  Reusing reordering from past factorization\n");CHKERRQ(ierr);}
    if (factor->factortype == MAT_FACTOR_ILU || factor->factortype == MAT_FACTOR_ICC) {
      if (factor->info.dt > 0) {
        ierr = PetscViewerASCIIPrintf(viewer,"  drop tolerance %g\n",(double)factor->info.dt);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  max nonzeros per row %D\n",factor->info.dtcount);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  column permutation tolerance %g\n",(double)factor->info.dtcol);CHKERRQ(ierr);
      } else if (factor->info.levels == 1) {
        ierr = PetscViewerASCIIPrintf(viewer,"  %D level of fill\n",(PetscInt)factor->info.levels);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"  %D levels of fill\n",(PetscInt)factor->info.levels);CHKERRQ(ierr);
      }
    }

    ierr = PetscViewerASCIIPrintf(viewer,"  tolerance for zero pivot %g\n",(double)factor->info.zeropivot);CHKERRQ(ierr);
    if (MatFactorShiftTypesDetail[(int)factor->info.shifttype]) { /* Only print when using a nontrivial shift */
      ierr = PetscViewerASCIIPrintf(viewer,"  using %s [%s]\n",MatFactorShiftTypesDetail[(int)factor->info.shifttype],MatFactorShiftTypes[(int)factor->info.shifttype]);CHKERRQ(ierr);
    }

    ierr = PetscStrcmp(factor->ordering,MATORDERINGNATURAL_OR_ND,&flg);CHKERRQ(ierr);
    if (flg) {
      PetscBool isseqsbaij;
      ierr = PetscObjectTypeCompareAny((PetscObject)pc->pmat,&isseqsbaij,MATSEQSBAIJ,MATSEQBAIJ,NULL);CHKERRQ(ierr);
      if (isseqsbaij) {
        ordering = MATORDERINGNATURAL;
      } else {
        ordering = MATORDERINGND;
      }
    } else {
      ordering = factor->ordering;
    }
    if (!factor->fact) {
      ierr = PetscViewerASCIIPrintf(viewer,"  matrix ordering: %s (may be overridden during setup)\n",ordering);CHKERRQ(ierr);
    } else {
      PetscBool useordering;
      MatInfo   info;
      ierr = MatFactorGetUseOrdering(factor->fact,&useordering);CHKERRQ(ierr);
      if (!useordering) ordering = "external";
      ierr = PetscViewerASCIIPrintf(viewer,"  matrix ordering: %s\n",ordering);CHKERRQ(ierr);
      ierr = MatGetInfo(factor->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  factor fill ratio given %g, needed %g\n",(double)info.fill_ratio_given,(double)info.fill_ratio_needed);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"    Factored matrix follows:\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
      ierr = MatView(factor->fact,viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }

  } else if (isstring) {
    MatFactorType t;
    ierr = MatGetFactorType(factor->fact,&t);CHKERRQ(ierr);
    if (t == MAT_FACTOR_ILU || t == MAT_FACTOR_ICC) {
      ierr = PetscViewerStringSPrintf(viewer," lvls=%D,order=%s",(PetscInt)factor->info.levels,factor->ordering);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
