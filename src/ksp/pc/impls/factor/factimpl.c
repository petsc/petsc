
#include <../src/ksp/pc/impls/factor/factor.h> /*I "petscpc.h"  I*/

PetscErrorCode PCFactorSetUpMatSolverType_Factor(PC pc)
{
  PC_Factor *icc = (PC_Factor *)pc->data;

  PetscFunctionBegin;
  PetscCheck(pc->pmat, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "You can only call this routine after the matrix object has been provided to the solver, for example with KSPSetOperators() or SNESSetJacobian()");
  if (!pc->setupcalled && !((PC_Factor *)icc)->fact) PetscCall(MatGetFactor(pc->pmat, ((PC_Factor *)icc)->solvertype, ((PC_Factor *)icc)->factortype, &((PC_Factor *)icc)->fact));
  PetscFunctionReturn(0);
}

PetscErrorCode PCFactorSetZeroPivot_Factor(PC pc, PetscReal z)
{
  PC_Factor *ilu = (PC_Factor *)pc->data;

  PetscFunctionBegin;
  ilu->info.zeropivot = z;
  PetscFunctionReturn(0);
}

PetscErrorCode PCFactorSetShiftType_Factor(PC pc, MatFactorShiftType shifttype)
{
  PC_Factor *dir = (PC_Factor *)pc->data;

  PetscFunctionBegin;
  if (shifttype == (MatFactorShiftType)PETSC_DECIDE) dir->info.shifttype = (PetscReal)MAT_SHIFT_NONE;
  else {
    dir->info.shifttype = (PetscReal)shifttype;
    if ((shifttype == MAT_SHIFT_NONZERO || shifttype == MAT_SHIFT_INBLOCKS) && dir->info.shiftamount == 0.0) { dir->info.shiftamount = 100.0 * PETSC_MACHINE_EPSILON; /* set default amount if user has not called PCFactorSetShiftAmount() yet */ }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCFactorSetShiftAmount_Factor(PC pc, PetscReal shiftamount)
{
  PC_Factor *dir = (PC_Factor *)pc->data;

  PetscFunctionBegin;
  if (shiftamount == (PetscReal)PETSC_DECIDE) dir->info.shiftamount = 100.0 * PETSC_MACHINE_EPSILON;
  else dir->info.shiftamount = shiftamount;
  PetscFunctionReturn(0);
}

PetscErrorCode PCFactorSetDropTolerance_Factor(PC pc, PetscReal dt, PetscReal dtcol, PetscInt dtcount)
{
  PC_Factor *ilu = (PC_Factor *)pc->data;

  PetscFunctionBegin;
  if (pc->setupcalled && (!ilu->info.usedt || ((PC_Factor *)ilu)->info.dt != dt || ((PC_Factor *)ilu)->info.dtcol != dtcol || ((PC_Factor *)ilu)->info.dtcount != dtcount)) {
    SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "Cannot change tolerance after use");
  }
  ilu->info.usedt   = PETSC_TRUE;
  ilu->info.dt      = dt;
  ilu->info.dtcol   = dtcol;
  ilu->info.dtcount = dtcount;
  ilu->info.fill    = PETSC_DEFAULT;
  PetscFunctionReturn(0);
}

PetscErrorCode PCFactorSetFill_Factor(PC pc, PetscReal fill)
{
  PC_Factor *dir = (PC_Factor *)pc->data;

  PetscFunctionBegin;
  dir->info.fill = fill;
  PetscFunctionReturn(0);
}

PetscErrorCode PCFactorSetMatOrderingType_Factor(PC pc, MatOrderingType ordering)
{
  PC_Factor *dir = (PC_Factor *)pc->data;
  PetscBool  flg;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    PetscCall(PetscFree(dir->ordering));
    PetscCall(PetscStrallocpy(ordering, (char **)&dir->ordering));
  } else {
    PetscCall(PetscStrcmp(dir->ordering, ordering, &flg));
    PetscCheck(flg, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "Cannot change ordering after use");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCFactorGetLevels_Factor(PC pc, PetscInt *levels)
{
  PC_Factor *ilu = (PC_Factor *)pc->data;

  PetscFunctionBegin;
  *levels = ilu->info.levels;
  PetscFunctionReturn(0);
}

PetscErrorCode PCFactorGetZeroPivot_Factor(PC pc, PetscReal *pivot)
{
  PC_Factor *ilu = (PC_Factor *)pc->data;

  PetscFunctionBegin;
  *pivot = ilu->info.zeropivot;
  PetscFunctionReturn(0);
}

PetscErrorCode PCFactorGetShiftAmount_Factor(PC pc, PetscReal *shift)
{
  PC_Factor *ilu = (PC_Factor *)pc->data;

  PetscFunctionBegin;
  *shift = ilu->info.shiftamount;
  PetscFunctionReturn(0);
}

PetscErrorCode PCFactorGetShiftType_Factor(PC pc, MatFactorShiftType *type)
{
  PC_Factor *ilu = (PC_Factor *)pc->data;

  PetscFunctionBegin;
  *type = (MatFactorShiftType)(int)ilu->info.shifttype;
  PetscFunctionReturn(0);
}

PetscErrorCode PCFactorSetLevels_Factor(PC pc, PetscInt levels)
{
  PC_Factor *ilu = (PC_Factor *)pc->data;

  PetscFunctionBegin;
  if (!pc->setupcalled) ilu->info.levels = levels;
  else if (ilu->info.levels != levels) {
    PetscUseTypeMethod(pc, reset); /* remove previous factored matrices */
    pc->setupcalled  = 0;          /* force a complete rebuild of preconditioner factored matrices */
    ilu->info.levels = levels;
  } else PetscCheck(!ilu->info.usedt, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "Cannot change levels after use with ILUdt");
  PetscFunctionReturn(0);
}

PetscErrorCode PCFactorSetAllowDiagonalFill_Factor(PC pc, PetscBool flg)
{
  PC_Factor *dir = (PC_Factor *)pc->data;

  PetscFunctionBegin;
  dir->info.diagonal_fill = (PetscReal)flg;
  PetscFunctionReturn(0);
}

PetscErrorCode PCFactorGetAllowDiagonalFill_Factor(PC pc, PetscBool *flg)
{
  PC_Factor *dir = (PC_Factor *)pc->data;

  PetscFunctionBegin;
  *flg = dir->info.diagonal_fill ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PCFactorSetPivotInBlocks_Factor(PC pc, PetscBool pivot)
{
  PC_Factor *dir = (PC_Factor *)pc->data;

  PetscFunctionBegin;
  dir->info.pivotinblocks = pivot ? 1.0 : 0.0;
  PetscFunctionReturn(0);
}

PetscErrorCode PCFactorGetMatrix_Factor(PC pc, Mat *mat)
{
  PC_Factor *ilu = (PC_Factor *)pc->data;

  PetscFunctionBegin;
  PetscCheck(ilu->fact, PetscObjectComm((PetscObject)pc), PETSC_ERR_ORDER, "Matrix not yet factored; call after KSPSetUp() or PCSetUp()");
  *mat = ilu->fact;
  PetscFunctionReturn(0);
}

/* allow access to preallocation information */
#include <petsc/private/matimpl.h>

PetscErrorCode PCFactorSetMatSolverType_Factor(PC pc, MatSolverType stype)
{
  PC_Factor *lu = (PC_Factor *)pc->data;

  PetscFunctionBegin;
  if (lu->fact && lu->fact->assembled) {
    MatSolverType ltype;
    PetscBool     flg;
    PetscCall(MatFactorGetSolverType(lu->fact, &ltype));
    PetscCall(PetscStrcmp(stype, ltype, &flg));
    PetscCheck(flg, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "Cannot change solver matrix package from %s to %s after PC has been setup or used", ltype, stype);
  }

  PetscCall(PetscFree(lu->solvertype));
  PetscCall(PetscStrallocpy(stype, &lu->solvertype));
  PetscFunctionReturn(0);
}

PetscErrorCode PCFactorGetMatSolverType_Factor(PC pc, MatSolverType *stype)
{
  PC_Factor *lu = (PC_Factor *)pc->data;

  PetscFunctionBegin;
  *stype = lu->solvertype;
  PetscFunctionReturn(0);
}

PetscErrorCode PCFactorSetColumnPivot_Factor(PC pc, PetscReal dtcol)
{
  PC_Factor *dir = (PC_Factor *)pc->data;

  PetscFunctionBegin;
  PetscCheck(dtcol >= 0.0 && dtcol <= 1.0, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_OUTOFRANGE, "Column pivot tolerance is %g must be between 0 and 1", (double)dtcol);
  dir->info.dtcol = dtcol;
  PetscFunctionReturn(0);
}

PetscErrorCode PCSetFromOptions_Factor(PC pc, PetscOptionItems *PetscOptionsObject)
{
  PC_Factor        *factor = (PC_Factor *)pc->data;
  PetscBool         flg, set;
  char              tname[256], solvertype[64];
  PetscFunctionList ordlist;
  PetscEnum         etmp;
  PetscBool         inplace;

  PetscFunctionBegin;
  PetscCall(PCFactorGetUseInPlace(pc, &inplace));
  PetscCall(PetscOptionsBool("-pc_factor_in_place", "Form factored matrix in the same memory as the matrix", "PCFactorSetUseInPlace", inplace, &flg, &set));
  if (set) PetscCall(PCFactorSetUseInPlace(pc, flg));
  PetscCall(PetscOptionsReal("-pc_factor_fill", "Expected non-zeros in factored matrix", "PCFactorSetFill", ((PC_Factor *)factor)->info.fill, &((PC_Factor *)factor)->info.fill, NULL));

  PetscCall(PetscOptionsEnum("-pc_factor_shift_type", "Type of shift to add to diagonal", "PCFactorSetShiftType", MatFactorShiftTypes, (PetscEnum)(int)((PC_Factor *)factor)->info.shifttype, &etmp, &flg));
  if (flg) PetscCall(PCFactorSetShiftType(pc, (MatFactorShiftType)etmp));
  PetscCall(PetscOptionsReal("-pc_factor_shift_amount", "Shift added to diagonal", "PCFactorSetShiftAmount", ((PC_Factor *)factor)->info.shiftamount, &((PC_Factor *)factor)->info.shiftamount, NULL));

  PetscCall(PetscOptionsReal("-pc_factor_zeropivot", "Pivot is considered zero if less than", "PCFactorSetZeroPivot", ((PC_Factor *)factor)->info.zeropivot, &((PC_Factor *)factor)->info.zeropivot, NULL));
  PetscCall(PetscOptionsReal("-pc_factor_column_pivot", "Column pivot tolerance (used only for some factorization)", "PCFactorSetColumnPivot", ((PC_Factor *)factor)->info.dtcol, &((PC_Factor *)factor)->info.dtcol, &flg));

  PetscCall(PetscOptionsBool("-pc_factor_pivot_in_blocks", "Pivot inside matrix dense blocks for BAIJ and SBAIJ", "PCFactorSetPivotInBlocks", ((PC_Factor *)factor)->info.pivotinblocks ? PETSC_TRUE : PETSC_FALSE, &flg, &set));
  if (set) PetscCall(PCFactorSetPivotInBlocks(pc, flg));

  PetscCall(PetscOptionsBool("-pc_factor_reuse_fill", "Use fill from previous factorization", "PCFactorSetReuseFill", PETSC_FALSE, &flg, &set));
  if (set) PetscCall(PCFactorSetReuseFill(pc, flg));
  PetscCall(PetscOptionsBool("-pc_factor_reuse_ordering", "Reuse ordering from previous factorization", "PCFactorSetReuseOrdering", PETSC_FALSE, &flg, &set));
  if (set) PetscCall(PCFactorSetReuseOrdering(pc, flg));

  PetscCall(PetscOptionsDeprecated("-pc_factor_mat_solver_package", "-pc_factor_mat_solver_type", "3.9", NULL));
  PetscCall(PetscOptionsString("-pc_factor_mat_solver_type", "Specific direct solver to use", "MatGetFactor", ((PC_Factor *)factor)->solvertype, solvertype, sizeof(solvertype), &flg));
  if (flg) PetscCall(PCFactorSetMatSolverType(pc, solvertype));
  PetscCall(PCFactorSetDefaultOrdering_Factor(pc));
  PetscCall(MatGetOrderingList(&ordlist));
  PetscCall(PetscOptionsFList("-pc_factor_mat_ordering_type", "Reordering to reduce nonzeros in factored matrix", "PCFactorSetMatOrderingType", ordlist, ((PC_Factor *)factor)->ordering, tname, sizeof(tname), &flg));
  if (flg) PetscCall(PCFactorSetMatOrderingType(pc, tname));
  PetscFunctionReturn(0);
}

PetscErrorCode PCView_Factor(PC pc, PetscViewer viewer)
{
  PC_Factor        *factor = (PC_Factor *)pc->data;
  PetscBool         isstring, iascii, canuseordering;
  MatInfo           info;
  MatOrderingType   ordering;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSTRING, &isstring));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    if (factor->inplace) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  in-place factorization\n"));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  out-of-place factorization\n"));
    }

    if (factor->reusefill) PetscCall(PetscViewerASCIIPrintf(viewer, "  Reusing fill from past factorization\n"));
    if (factor->reuseordering) PetscCall(PetscViewerASCIIPrintf(viewer, "  Reusing reordering from past factorization\n"));
    if (factor->factortype == MAT_FACTOR_ILU || factor->factortype == MAT_FACTOR_ICC) {
      if (factor->info.dt > 0) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "  drop tolerance %g\n", (double)factor->info.dt));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  max nonzeros per row %" PetscInt_FMT "\n", (PetscInt)factor->info.dtcount));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  column permutation tolerance %g\n", (double)factor->info.dtcol));
      } else if (factor->info.levels == 1) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "  %" PetscInt_FMT " level of fill\n", (PetscInt)factor->info.levels));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "  %" PetscInt_FMT " levels of fill\n", (PetscInt)factor->info.levels));
      }
    }

    PetscCall(PetscViewerASCIIPrintf(viewer, "  tolerance for zero pivot %g\n", (double)factor->info.zeropivot));
    if (MatFactorShiftTypesDetail[(int)factor->info.shifttype]) { /* Only print when using a nontrivial shift */
      PetscCall(PetscViewerASCIIPrintf(viewer, "  using %s [%s]\n", MatFactorShiftTypesDetail[(int)factor->info.shifttype], MatFactorShiftTypes[(int)factor->info.shifttype]));
    }

    if (factor->fact) {
      PetscCall(MatFactorGetCanUseOrdering(factor->fact, &canuseordering));
      if (!canuseordering) ordering = MATORDERINGEXTERNAL;
      else ordering = factor->ordering;
      PetscCall(PetscViewerASCIIPrintf(viewer, "  matrix ordering: %s\n", ordering));
      if (!factor->fact->assembled) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "  matrix solver type: %s\n", factor->fact->solvertype));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  matrix not yet factored; no additional information available\n"));
      } else {
        PetscCall(MatGetInfo(factor->fact, MAT_LOCAL, &info));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  factor fill ratio given %g, needed %g\n", (double)info.fill_ratio_given, (double)info.fill_ratio_needed));
        PetscCall(PetscViewerASCIIPrintf(viewer, "    Factored matrix follows:\n"));
        PetscCall(PetscViewerASCIIPushTab(viewer));
        PetscCall(PetscViewerASCIIPushTab(viewer));
        PetscCall(PetscViewerASCIIPushTab(viewer));
        PetscCall(PetscViewerGetFormat(viewer, &format));
        PetscCall(PetscViewerPushFormat(viewer, format != PETSC_VIEWER_ASCII_INFO_DETAIL ? PETSC_VIEWER_ASCII_INFO : PETSC_VIEWER_ASCII_INFO_DETAIL));
        PetscCall(MatView(factor->fact, viewer));
        PetscCall(PetscViewerPopFormat(viewer));
        PetscCall(PetscViewerASCIIPopTab(viewer));
        PetscCall(PetscViewerASCIIPopTab(viewer));
        PetscCall(PetscViewerASCIIPopTab(viewer));
      }
    }

  } else if (isstring) {
    MatFactorType t;
    PetscCall(MatGetFactorType(factor->fact, &t));
    if (t == MAT_FACTOR_ILU || t == MAT_FACTOR_ICC) PetscCall(PetscViewerStringSPrintf(viewer, " lvls=%" PetscInt_FMT ",order=%s", (PetscInt)factor->info.levels, factor->ordering));
  }
  PetscFunctionReturn(0);
}
