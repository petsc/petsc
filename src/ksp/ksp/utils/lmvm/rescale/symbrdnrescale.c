#include <petscdevice.h>
#include "symbrdnrescale.h"

PetscLogEvent SBRDN_Rescale;

const char *const MatLMVMSymBroydenScaleTypes[] = {"none", "scalar", "diagonal", "user", "decide", "MatLMVMSymBroydenScaleType", "MAT_LMVM_SYMBROYDEN_SCALING_", NULL};

PETSC_INTERN PetscErrorCode SymBroydenRescaleUpdate(Mat B, SymBroydenRescale ldb)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "not implemented");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleSetDelta(Mat B, SymBroydenRescale ldb, PetscReal delta)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool same;

  PetscFunctionBegin;
  same       = (delta == ldb->delta) ? PETSC_TRUE : PETSC_FALSE;
  ldb->delta = delta;
  ldb->delta = PetscMin(ldb->delta, ldb->delta_max);
  ldb->delta = PetscMax(ldb->delta, ldb->delta_min);
  // if there have been no updates yet, propagate delta to J0
  if (!same && !lmvm->prev_set && (ldb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_SCALAR || ldb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL)) {
    ldb->initialized = PETSC_FALSE;
    B->preallocated  = PETSC_FALSE; // make sure SyBroydenInitializeJ0() is called in the next MatCheckPreallocated()
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "not implemented");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleCopy(SymBroydenRescale bctx, SymBroydenRescale mctx)
{
  PetscFunctionBegin;
  mctx->scale_type = bctx->scale_type;
  mctx->theta      = bctx->theta;
  mctx->alpha      = bctx->alpha;
  mctx->beta       = bctx->beta;
  mctx->rho        = bctx->rho;
  mctx->delta      = bctx->delta;
  mctx->delta_min  = bctx->delta_min;
  mctx->delta_max  = bctx->delta_max;
  mctx->tol        = bctx->tol;
  mctx->sigma_hist = bctx->sigma_hist;
  mctx->forward    = bctx->forward;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleSetDiagonalMode(SymBroydenRescale ldb, PetscBool forward)
{
  PetscFunctionBegin;
  ldb->forward = forward;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleGetType(SymBroydenRescale ldb, MatLMVMSymBroydenScaleType *stype)
{
  PetscFunctionBegin;
  *stype = ldb->scale_type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleSetType(SymBroydenRescale ldb, MatLMVMSymBroydenScaleType stype)
{
  PetscFunctionBegin;
  ldb->scale_type = stype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleSetFromOptions(Mat B, SymBroydenRescale ldb, PetscOptionItems PetscOptionsObject)
{
  MatLMVMSymBroydenScaleType stype = ldb->scale_type;
  PetscBool                  flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Restricted Broyden method for updating diagonal Jacobian approximation (MATLMVMDIAGBRDN)");
  PetscCall(PetscOptionsEnum("-mat_lmvm_scale_type", "(developer) scaling type applied to J0", "MatLMVMSymBroydenScaleType", MatLMVMSymBroydenScaleTypes, (PetscEnum)stype, (PetscEnum *)&stype, &flg));
  PetscCall(PetscOptionsReal("-mat_lmvm_theta", "(developer) convex ratio between BFGS and DFP components of the diagonal J0 scaling", "", ldb->theta, &ldb->theta, NULL));
  PetscCall(PetscOptionsReal("-mat_lmvm_rho", "(developer) update limiter in the J0 scaling", "", ldb->rho, &ldb->rho, NULL));
  PetscCall(PetscOptionsReal("-mat_lmvm_tol", "(developer) tolerance for bounding rescaling denominator", "", ldb->tol, &ldb->tol, NULL));
  PetscCall(PetscOptionsRangeReal("-mat_lmvm_alpha", "(developer) convex ratio in the J0 scaling", "", ldb->alpha, &ldb->alpha, NULL, 0.0, 1.0));
  PetscCall(PetscOptionsBool("-mat_lmvm_forward", "Forward -> Update diagonal scaling for B. Else -> diagonal scaling for H.", "", ldb->forward, &ldb->forward, NULL));
  PetscCall(PetscOptionsReal("-mat_lmvm_beta", "(developer) exponential factor in the diagonal J0 scaling", "", ldb->beta, &ldb->beta, NULL));
  PetscCall(PetscOptionsBoundedInt("-mat_lmvm_sigma_hist", "(developer) number of past updates to use in the default J0 scalar", "", ldb->sigma_hist, &ldb->sigma_hist, NULL, 0));
  PetscOptionsHeadEnd();
  PetscCheck(!(ldb->theta < 0.0) && !(ldb->theta > 1.0), PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio for the diagonal J0 scale cannot be outside the range of [0, 1]");
  PetscCheck(!(ldb->alpha < 0.0) && !(ldb->alpha > 1.0), PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio in the J0 scaling cannot be outside the range of [0, 1]");
  PetscCheck(!(ldb->rho < 0.0) && !(ldb->rho > 1.0), PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex update limiter in the J0 scaling cannot be outside the range of [0, 1]");
  PetscCheck(ldb->sigma_hist >= 0, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "J0 scaling history length cannot be negative");
  if (flg) PetscCall(SymBroydenRescaleSetType(ldb, stype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SymBroydenRescaleAllocate(Mat B, SymBroydenRescale ldb)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  if (!ldb->allocated) {
    PetscCall(PetscMalloc3(ldb->sigma_hist, &ldb->yty, ldb->sigma_hist, &ldb->yts, ldb->sigma_hist, &ldb->sts));
    PetscCall(VecDuplicate(lmvm->Xprev, &ldb->invDnew));
    PetscCall(VecDuplicate(lmvm->Xprev, &ldb->BFGS));
    PetscCall(VecDuplicate(lmvm->Xprev, &ldb->DFP));
    PetscCall(VecDuplicate(lmvm->Xprev, &ldb->U));
    PetscCall(VecDuplicate(lmvm->Xprev, &ldb->V));
    PetscCall(VecDuplicate(lmvm->Xprev, &ldb->W));
    ldb->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleSetUp(Mat B, SymBroydenRescale ldb)
{
  PetscFunctionBegin;
  PetscCall(SymBroydenRescaleAllocate(B, ldb));
  if (ldb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DECIDE) {
    Mat       J0;
    PetscBool is_constant_or_diagonal;

    PetscCall(MatLMVMGetJ0(B, &J0));
    PetscCall(PetscObjectTypeCompareAny((PetscObject)J0, &is_constant_or_diagonal, MATCONSTANTDIAGONAL, MATDIAGONAL, ""));
    ldb->scale_type = is_constant_or_diagonal ? MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL : MAT_LMVM_SYMBROYDEN_SCALE_NONE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleInitializeJ0(Mat B, SymBroydenRescale ldb)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "not implemented");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleView(SymBroydenRescale ldb, PetscViewer pv)
{
  PetscFunctionBegin;
  PetscBool isascii;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(pv, "Rescale type: %s\n", MatLMVMSymBroydenScaleTypes[ldb->scale_type]));
    if (ldb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_SCALAR || ldb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) {
      PetscCall(PetscViewerASCIIPrintf(pv, "Rescale history: %" PetscInt_FMT "\n", ldb->sigma_hist));
      PetscCall(PetscViewerASCIIPrintf(pv, "Rescale params: alpha=%g, beta=%g, rho=%g\n", (double)ldb->alpha, (double)ldb->beta, (double)ldb->rho));
    }
    if (ldb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) PetscCall(PetscViewerASCIIPrintf(pv, "Rescale convex factor: theta=%g\n", (double)ldb->theta));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleReset(Mat B, SymBroydenRescale ldb, PetscBool destructive)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "not implemented");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleDestroy(SymBroydenRescale *ldb)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "not implemented");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleCreate(SymBroydenRescale *ldb)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(ldb));
  (*ldb)->scale_type = MAT_LMVM_SYMBROYDEN_SCALE_DECIDE;
  (*ldb)->theta      = 0.0;
  (*ldb)->alpha      = 1.0;
  (*ldb)->rho        = 1.0;
  (*ldb)->forward    = PETSC_TRUE;
  (*ldb)->beta       = 0.5;
  (*ldb)->delta      = 1.0;
  (*ldb)->delta_min  = 1e-7;
  (*ldb)->delta_max  = 100.0;
  (*ldb)->tol        = 1e-8;
  (*ldb)->sigma_hist = 1;
  (*ldb)->allocated  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
