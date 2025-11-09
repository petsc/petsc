/*
    The PC (preconditioner) interface routines, callable by users.
*/
#include <petsc/private/pcimpl.h> /*I "petscksp.h" I*/
#include <petscdm.h>

/* Logging support */
PetscClassId  PC_CLASSID;
PetscLogEvent PC_SetUp, PC_SetUpOnBlocks, PC_Apply, PC_MatApply, PC_ApplyCoarse, PC_ApplySymmetricLeft;
PetscLogEvent PC_ApplySymmetricRight, PC_ModifySubMatrices, PC_ApplyOnBlocks, PC_ApplyTransposeOnBlocks;
PetscInt      PetscMGLevelId;
PetscLogStage PCMPIStage;

PETSC_INTERN PetscErrorCode PCGetDefaultType_Private(PC pc, const char *type[])
{
  PetscMPIInt size;
  PetscBool   hasopblock, hasopsolve, flg1, flg2, set, flg3, isnormal;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc), &size));
  if (pc->pmat) {
    PetscCall(MatHasOperation(pc->pmat, MATOP_GET_DIAGONAL_BLOCK, &hasopblock));
    PetscCall(MatHasOperation(pc->pmat, MATOP_SOLVE, &hasopsolve));
    if (size == 1) {
      PetscCall(MatGetFactorAvailable(pc->pmat, "petsc", MAT_FACTOR_ICC, &flg1));
      PetscCall(MatGetFactorAvailable(pc->pmat, "petsc", MAT_FACTOR_ILU, &flg2));
      PetscCall(MatIsSymmetricKnown(pc->pmat, &set, &flg3));
      PetscCall(PetscObjectTypeCompareAny((PetscObject)pc->pmat, &isnormal, MATNORMAL, MATNORMALHERMITIAN, NULL));
      if (flg1 && (!flg2 || (set && flg3))) {
        *type = PCICC;
      } else if (flg2) {
        *type = PCILU;
      } else if (isnormal) {
        *type = PCNONE;
      } else if (hasopblock) { /* likely is a parallel matrix run on one processor */
        *type = PCBJACOBI;
      } else if (hasopsolve) {
        *type = PCMAT;
      } else {
        *type = PCNONE;
      }
    } else {
      if (hasopblock) {
        *type = PCBJACOBI;
      } else if (hasopsolve) {
        *type = PCMAT;
      } else {
        *type = PCNONE;
      }
    }
  } else *type = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* do not log solves, setup and applications of preconditioners while constructing preconditioners; perhaps they should be logged separately from the regular solves */
PETSC_EXTERN PetscLogEvent KSP_Solve, KSP_SetUp;

static PetscErrorCode PCLogEventsDeactivatePush(void)
{
  PetscFunctionBegin;
  PetscCall(KSPInitializePackage());
  PetscCall(PetscLogEventDeactivatePush(KSP_Solve));
  PetscCall(PetscLogEventDeactivatePush(KSP_SetUp));
  PetscCall(PetscLogEventDeactivatePush(PC_Apply));
  PetscCall(PetscLogEventDeactivatePush(PC_SetUp));
  PetscCall(PetscLogEventDeactivatePush(PC_SetUpOnBlocks));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCLogEventsDeactivatePop(void)
{
  PetscFunctionBegin;
  PetscCall(KSPInitializePackage());
  PetscCall(PetscLogEventDeactivatePop(KSP_Solve));
  PetscCall(PetscLogEventDeactivatePop(KSP_SetUp));
  PetscCall(PetscLogEventDeactivatePop(PC_Apply));
  PetscCall(PetscLogEventDeactivatePop(PC_SetUp));
  PetscCall(PetscLogEventDeactivatePop(PC_SetUpOnBlocks));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCReset - Resets a `PC` context to the state it was in before `PCSetUp()` was called, and removes any allocated `Vec` and `Mat` from its data structure

  Collective

  Input Parameter:
. pc - the `PC` preconditioner context

  Level: developer

  Notes:
  Any options set, including those set with `KSPSetFromOptions()` remain.

  This allows a `PC` to be reused for a different sized linear system but using the same options that have been previously set in `pc`

.seealso: [](ch_ksp), `PC`, `PCCreate()`, `PCSetUp()`
@*/
PetscErrorCode PCReset(PC pc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryTypeMethod(pc, reset);
  PetscCall(VecDestroy(&pc->diagonalscaleright));
  PetscCall(VecDestroy(&pc->diagonalscaleleft));
  PetscCall(MatDestroy(&pc->pmat));
  PetscCall(MatDestroy(&pc->mat));

  pc->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCDestroy - Destroys `PC` context that was created with `PCCreate()`.

  Collective

  Input Parameter:
. pc - the `PC` preconditioner context

  Level: developer

.seealso: [](ch_ksp), `PC`, `PCCreate()`, `PCSetUp()`
@*/
PetscErrorCode PCDestroy(PC *pc)
{
  PetscFunctionBegin;
  if (!*pc) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*pc, PC_CLASSID, 1);
  if (--((PetscObject)*pc)->refct > 0) {
    *pc = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PCReset(*pc));

  /* if memory was published with SAWs then destroy it */
  PetscCall(PetscObjectSAWsViewOff((PetscObject)*pc));
  PetscTryTypeMethod(*pc, destroy);
  PetscCall(DMDestroy(&(*pc)->dm));
  PetscCall(PetscHeaderDestroy(pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCGetDiagonalScale - Indicates if the preconditioner applies an additional left and right
  scaling as needed by certain time-stepping codes.

  Logically Collective

  Input Parameter:
. pc - the `PC` preconditioner context

  Output Parameter:
. flag - `PETSC_TRUE` if it applies the scaling

  Level: developer

  Note:
  If this returns `PETSC_TRUE` then the system solved via the Krylov method is, for left and right preconditioning,

  $$
  \begin{align*}
  D M A D^{-1} y = D M b  \\
  D A M D^{-1} z = D b.
  \end{align*}
  $$

.seealso: [](ch_ksp), `PC`, `PCCreate()`, `PCSetUp()`, `PCDiagonalScaleLeft()`, `PCDiagonalScaleRight()`, `PCSetDiagonalScale()`
@*/
PetscErrorCode PCGetDiagonalScale(PC pc, PetscBool *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscAssertPointer(flag, 2);
  *flag = pc->diagonalscale;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCSetDiagonalScale - Indicates the left scaling to use to apply an additional left and right
  scaling as needed by certain time-stepping codes.

  Logically Collective

  Input Parameters:
+ pc - the `PC` preconditioner context
- s  - scaling vector

  Level: intermediate

  Notes:
  The system solved via the Krylov method is, for left and right preconditioning,
  $$
  \begin{align*}
  D M A D^{-1} y = D M b \\
  D A M D^{-1} z = D b.
  \end{align*}
  $$

  `PCDiagonalScaleLeft()` scales a vector by $D$. `PCDiagonalScaleRight()` scales a vector by $D^{-1}$.

.seealso: [](ch_ksp), `PCCreate()`, `PCSetUp()`, `PCDiagonalScaleLeft()`, `PCDiagonalScaleRight()`, `PCGetDiagonalScale()`
@*/
PetscErrorCode PCSetDiagonalScale(PC pc, Vec s)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(s, VEC_CLASSID, 2);
  pc->diagonalscale = PETSC_TRUE;

  PetscCall(PetscObjectReference((PetscObject)s));
  PetscCall(VecDestroy(&pc->diagonalscaleleft));

  pc->diagonalscaleleft = s;

  PetscCall(VecDuplicate(s, &pc->diagonalscaleright));
  PetscCall(VecCopy(s, pc->diagonalscaleright));
  PetscCall(VecReciprocal(pc->diagonalscaleright));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCDiagonalScaleLeft - Scales a vector by the left scaling as needed by certain time-stepping codes.

  Logically Collective

  Input Parameters:
+ pc  - the `PC` preconditioner context
. in  - input vector
- out - scaled vector (maybe the same as in)

  Level: intermediate

  Notes:
  The system solved via the Krylov method is, for left and right preconditioning,

  $$
  \begin{align*}
  D M A D^{-1} y = D M b  \\
  D A M D^{-1} z = D b.
  \end{align*}
  $$

  `PCDiagonalScaleLeft()` scales a vector by $D$. `PCDiagonalScaleRight()` scales a vector by $D^{-1}$.

  If diagonal scaling is turned off and `in` is not `out` then `in` is copied to `out`

.seealso: [](ch_ksp), `PCCreate()`, `PCSetUp()`, `PCSetDiagonalScale()`, `PCDiagonalScaleRight()`, `MatDiagonalScale()`
@*/
PetscErrorCode PCDiagonalScaleLeft(PC pc, Vec in, Vec out)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(in, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(out, VEC_CLASSID, 3);
  if (pc->diagonalscale) {
    PetscCall(VecPointwiseMult(out, pc->diagonalscaleleft, in));
  } else if (in != out) {
    PetscCall(VecCopy(in, out));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCDiagonalScaleRight - Scales a vector by the right scaling as needed by certain time-stepping codes.

  Logically Collective

  Input Parameters:
+ pc  - the `PC` preconditioner context
. in  - input vector
- out - scaled vector (maybe the same as in)

  Level: intermediate

  Notes:
  The system solved via the Krylov method is, for left and right preconditioning,

  $$
  \begin{align*}
  D M A D^{-1} y = D M b  \\
  D A M D^{-1} z = D b.
  \end{align*}
  $$

  `PCDiagonalScaleLeft()` scales a vector by $D$. `PCDiagonalScaleRight()` scales a vector by $D^{-1}$.

  If diagonal scaling is turned off and `in` is not `out` then `in` is copied to `out`

.seealso: [](ch_ksp), `PCCreate()`, `PCSetUp()`, `PCDiagonalScaleLeft()`, `PCSetDiagonalScale()`, `MatDiagonalScale()`
@*/
PetscErrorCode PCDiagonalScaleRight(PC pc, Vec in, Vec out)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(in, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(out, VEC_CLASSID, 3);
  if (pc->diagonalscale) {
    PetscCall(VecPointwiseMult(out, pc->diagonalscaleright, in));
  } else if (in != out) {
    PetscCall(VecCopy(in, out));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCSetUseAmat - Sets a flag to indicate that when the preconditioner needs to apply (part of) the
  operator during the preconditioning process it applies the Amat provided to `TSSetRHSJacobian()`,
  `TSSetIJacobian()`, `SNESSetJacobian()`, `KSPSetOperators()` or `PCSetOperators()` not the Pmat.

  Logically Collective

  Input Parameters:
+ pc  - the `PC` preconditioner context
- flg - `PETSC_TRUE` to use the Amat, `PETSC_FALSE` to use the Pmat (default is false)

  Options Database Key:
. -pc_use_amat <true,false> - use the amat argument to `KSPSetOperators()` or `PCSetOperators()` to apply the operator

  Level: intermediate

  Note:
  For the common case in which the linear system matrix and the matrix used to construct the
  preconditioner are identical, this routine has no affect.

.seealso: [](ch_ksp), `PC`, `PCGetUseAmat()`, `PCBJACOBI`, `PCMG`, `PCFIELDSPLIT`, `PCCOMPOSITE`,
          `KSPSetOperators()`, `PCSetOperators()`
@*/
PetscErrorCode PCSetUseAmat(PC pc, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  pc->useAmat = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCSetErrorIfFailure - Causes `PC` to generate an error if a floating point exception, for example a zero pivot, is detected.

  Logically Collective

  Input Parameters:
+ pc  - iterative context obtained from `PCCreate()`
- flg - `PETSC_TRUE` indicates you want the error generated

  Level: advanced

  Notes:
  Normally PETSc continues if a linear solver fails due to a failed setup of a preconditioner, you can call `KSPGetConvergedReason()` after a `KSPSolve()`
  to determine if it has converged or failed. Or use -ksp_error_if_not_converged to cause the program to terminate as soon as lack of convergence is
  detected.

  This is propagated into `KSP`s used by this `PC`, which then propagate it into `PC`s used by those `KSP`s

.seealso: [](ch_ksp), `PC`, `KSPSetErrorIfNotConverged()`, `PCGetInitialGuessNonzero()`, `PCSetInitialGuessKnoll()`, `PCGetInitialGuessKnoll()`
@*/
PetscErrorCode PCSetErrorIfFailure(PC pc, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidLogicalCollectiveBool(pc, flg, 2);
  pc->erroriffailure = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCGetUseAmat - Gets a flag to indicate that when the preconditioner needs to apply (part of) the
  operator during the preconditioning process it applies the Amat provided to `TSSetRHSJacobian()`,
  `TSSetIJacobian()`, `SNESSetJacobian()`, `KSPSetOperators()` or `PCSetOperators()` not the Pmat.

  Logically Collective

  Input Parameter:
. pc - the `PC` preconditioner context

  Output Parameter:
. flg - `PETSC_TRUE` to use the Amat, `PETSC_FALSE` to use the Pmat (default is false)

  Level: intermediate

  Note:
  For the common case in which the linear system matrix and the matrix used to construct the
  preconditioner are identical, this routine is does nothing.

.seealso: [](ch_ksp), `PC`, `PCSetUseAmat()`, `PCBJACOBI`, `PCMG`, `PCFIELDSPLIT`, `PCCOMPOSITE`
@*/
PetscErrorCode PCGetUseAmat(PC pc, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  *flg = pc->useAmat;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCSetKSPNestLevel - sets the amount of nesting the `KSP` that contains this `PC` has

  Collective

  Input Parameters:
+ pc    - the `PC`
- level - the nest level

  Level: developer

.seealso: [](ch_ksp), `KSPSetUp()`, `KSPSolve()`, `KSPDestroy()`, `KSP`, `KSPGMRES`, `KSPType`, `KSPGetNestLevel()`, `PCGetKSPNestLevel()`, `KSPSetNestLevel()`
@*/
PetscErrorCode PCSetKSPNestLevel(PC pc, PetscInt level)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidLogicalCollectiveInt(pc, level, 2);
  pc->kspnestlevel = level;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCGetKSPNestLevel - gets the amount of nesting the `KSP` that contains this `PC` has

  Not Collective

  Input Parameter:
. pc - the `PC`

  Output Parameter:
. level - the nest level

  Level: developer

.seealso: [](ch_ksp), `KSPSetUp()`, `KSPSolve()`, `KSPDestroy()`, `KSP`, `KSPGMRES`, `KSPType`, `KSPSetNestLevel()`, `PCSetKSPNestLevel()`, `KSPGetNestLevel()`
@*/
PetscErrorCode PCGetKSPNestLevel(PC pc, PetscInt *level)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscAssertPointer(level, 2);
  *level = pc->kspnestlevel;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCCreate - Creates a preconditioner context, `PC`

  Collective

  Input Parameter:
. comm - MPI communicator

  Output Parameter:
. newpc - location to put the `PC` preconditioner context

  Level: developer

  Notes:
  This is rarely called directly by users since `KSP` manages the `PC` objects it uses. Use `KSPGetPC()` to access the `PC` used by a `KSP`.

  Use `PCSetType()` or `PCSetFromOptions()` with the option `-pc_type pctype` to set the `PCType` for this `PC`

  The default preconditioner type `PCType` for sparse matrices is `PCILU` or `PCICC` with 0 fill on one process and block Jacobi (`PCBJACOBI`) with `PCILU` or `PCICC`
  in parallel. For dense matrices it is always `PCNONE`.

.seealso: [](ch_ksp), `PC`, `PCType`, `PCSetType`, `PCSetUp()`, `PCApply()`, `PCDestroy()`, `KSP`, `KSPGetPC()`
@*/
PetscErrorCode PCCreate(MPI_Comm comm, PC *newpc)
{
  PC pc;

  PetscFunctionBegin;
  PetscAssertPointer(newpc, 2);
  PetscCall(PCInitializePackage());

  PetscCall(PetscHeaderCreate(pc, PC_CLASSID, "PC", "Preconditioner", "PC", comm, PCDestroy, PCView));
  pc->mat                  = NULL;
  pc->pmat                 = NULL;
  pc->setupcalled          = PETSC_FALSE;
  pc->setfromoptionscalled = 0;
  pc->data                 = NULL;
  pc->diagonalscale        = PETSC_FALSE;
  pc->diagonalscaleleft    = NULL;
  pc->diagonalscaleright   = NULL;

  pc->modifysubmatrices  = NULL;
  pc->modifysubmatricesP = NULL;

  *newpc = pc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCApply - Applies the preconditioner to a vector.

  Collective

  Input Parameters:
+ pc - the `PC` preconditioner context
- x  - input vector

  Output Parameter:
. y - output vector

  Level: developer

.seealso: [](ch_ksp), `PC`, `PCApplyTranspose()`, `PCApplyBAorAB()`
@*/
PetscErrorCode PCApply(PC pc, Vec x, Vec y)
{
  PetscInt m, n, mv, nv;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 3);
  PetscCheck(x != y, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_IDN, "x and y must be different vectors");
  if (pc->erroriffailure) PetscCall(VecValidValues_Internal(x, 2, PETSC_TRUE));
  /* use pmat to check vector sizes since for KSPLSQR the pmat may be of a different size than mat */
  PetscCall(MatGetLocalSize(pc->pmat, &m, &n));
  PetscCall(VecGetLocalSize(x, &mv));
  PetscCall(VecGetLocalSize(y, &nv));
  /* check pmat * y = x is feasible */
  PetscCheck(mv == m, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Preconditioner number of local rows %" PetscInt_FMT " does not equal input vector size %" PetscInt_FMT, m, mv);
  PetscCheck(nv == n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Preconditioner number of local columns %" PetscInt_FMT " does not equal output vector size %" PetscInt_FMT, n, nv);
  PetscCall(VecSetErrorIfLocked(y, 3));

  PetscCall(PCSetUp(pc));
  PetscCall(VecLockReadPush(x));
  PetscCall(PetscLogEventBegin(PC_Apply, pc, x, y, 0));
  PetscUseTypeMethod(pc, apply, x, y);
  PetscCall(PetscLogEventEnd(PC_Apply, pc, x, y, 0));
  if (pc->erroriffailure) PetscCall(VecValidValues_Internal(y, 3, PETSC_FALSE));
  PetscCall(VecLockReadPop(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMatApplyTranspose_Private(PC pc, Mat X, Mat Y, PetscBool transpose)
{
  Mat       A;
  Vec       cy, cx;
  PetscInt  m1, M1, m2, M2, n1, N1, n2, N2, m3, M3, n3, N3;
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(X, MAT_CLASSID, 2);
  PetscValidHeaderSpecific(Y, MAT_CLASSID, 3);
  PetscCheckSameComm(pc, 1, X, 2);
  PetscCheckSameComm(pc, 1, Y, 3);
  PetscCheck(Y != X, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_IDN, "Y and X must be different matrices");
  PetscCall(PCGetOperators(pc, NULL, &A));
  PetscCall(MatGetLocalSize(A, &m3, &n3));
  PetscCall(MatGetLocalSize(X, &m2, &n2));
  PetscCall(MatGetLocalSize(Y, &m1, &n1));
  PetscCall(MatGetSize(A, &M3, &N3));
  PetscCall(MatGetSize(X, &M2, &N2));
  PetscCall(MatGetSize(Y, &M1, &N1));
  PetscCheck(n1 == n2 && N1 == N2, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible number of columns between block of input vectors (n,N) = (%" PetscInt_FMT ",%" PetscInt_FMT ") and block of output vectors (n,N) = (%" PetscInt_FMT ",%" PetscInt_FMT ")", n2, N2, n1, N1);
  PetscCheck(m2 == m3 && M2 == M3, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible layout between block of input vectors (m,M) = (%" PetscInt_FMT ",%" PetscInt_FMT ") and Pmat (m,M)x(n,N) = (%" PetscInt_FMT ",%" PetscInt_FMT ")x(%" PetscInt_FMT ",%" PetscInt_FMT ")", m2, M2, m3, M3, n3, N3);
  PetscCheck(m1 == n3 && M1 == N3, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible layout between block of output vectors (m,M) = (%" PetscInt_FMT ",%" PetscInt_FMT ") and Pmat (m,M)x(n,N) = (%" PetscInt_FMT ",%" PetscInt_FMT ")x(%" PetscInt_FMT ",%" PetscInt_FMT ")", m1, M1, m3, M3, n3, N3);
  PetscCall(PetscObjectBaseTypeCompareAny((PetscObject)Y, &match, MATSEQDENSE, MATMPIDENSE, ""));
  PetscCheck(match, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Provided block of output vectors not stored in a dense Mat");
  PetscCall(PetscObjectBaseTypeCompareAny((PetscObject)X, &match, MATSEQDENSE, MATMPIDENSE, ""));
  PetscCheck(match, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Provided block of input vectors not stored in a dense Mat");
  PetscCall(PCSetUp(pc));
  if (!transpose && pc->ops->matapply) {
    PetscCall(PetscLogEventBegin(PC_MatApply, pc, X, Y, 0));
    PetscUseTypeMethod(pc, matapply, X, Y);
    PetscCall(PetscLogEventEnd(PC_MatApply, pc, X, Y, 0));
  } else if (transpose && pc->ops->matapplytranspose) {
    PetscCall(PetscLogEventBegin(PC_MatApply, pc, X, Y, 0));
    PetscUseTypeMethod(pc, matapplytranspose, X, Y);
    PetscCall(PetscLogEventEnd(PC_MatApply, pc, X, Y, 0));
  } else {
    PetscCall(PetscInfo(pc, "PC type %s applying column by column\n", ((PetscObject)pc)->type_name));
    for (n1 = 0; n1 < N1; ++n1) {
      PetscCall(MatDenseGetColumnVecRead(X, n1, &cx));
      PetscCall(MatDenseGetColumnVecWrite(Y, n1, &cy));
      if (!transpose) PetscCall(PCApply(pc, cx, cy));
      else PetscCall(PCApplyTranspose(pc, cx, cy));
      PetscCall(MatDenseRestoreColumnVecWrite(Y, n1, &cy));
      PetscCall(MatDenseRestoreColumnVecRead(X, n1, &cx));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCMatApply - Applies the preconditioner to multiple vectors stored as a `MATDENSE`. Like `PCApply()`, `Y` and `X` must be different matrices.

  Collective

  Input Parameters:
+ pc - the `PC` preconditioner context
- X  - block of input vectors

  Output Parameter:
. Y - block of output vectors

  Level: developer

.seealso: [](ch_ksp), `PC`, `PCApply()`, `KSPMatSolve()`
@*/
PetscErrorCode PCMatApply(PC pc, Mat X, Mat Y)
{
  PetscFunctionBegin;
  PetscCall(PCMatApplyTranspose_Private(pc, X, Y, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCMatApplyTranspose - Applies the transpose of preconditioner to multiple vectors stored as a `MATDENSE`. Like `PCApplyTranspose()`, `Y` and `X` must be different matrices.

  Collective

  Input Parameters:
+ pc - the `PC` preconditioner context
- X  - block of input vectors

  Output Parameter:
. Y - block of output vectors

  Level: developer

.seealso: [](ch_ksp), `PC`, `PCApplyTranspose()`, `KSPMatSolveTranspose()`
@*/
PetscErrorCode PCMatApplyTranspose(PC pc, Mat X, Mat Y)
{
  PetscFunctionBegin;
  PetscCall(PCMatApplyTranspose_Private(pc, X, Y, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCApplySymmetricLeft - Applies the left part of a symmetric preconditioner to a vector.

  Collective

  Input Parameters:
+ pc - the `PC` preconditioner context
- x  - input vector

  Output Parameter:
. y - output vector

  Level: developer

  Note:
  Currently, this routine is implemented only for `PCICC` and `PCJACOBI` preconditioners.

.seealso: [](ch_ksp), `PC`, `PCApply()`, `PCApplySymmetricRight()`
@*/
PetscErrorCode PCApplySymmetricLeft(PC pc, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 3);
  PetscCheck(x != y, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_IDN, "x and y must be different vectors");
  if (pc->erroriffailure) PetscCall(VecValidValues_Internal(x, 2, PETSC_TRUE));
  PetscCall(PCSetUp(pc));
  PetscCall(VecLockReadPush(x));
  PetscCall(PetscLogEventBegin(PC_ApplySymmetricLeft, pc, x, y, 0));
  PetscUseTypeMethod(pc, applysymmetricleft, x, y);
  PetscCall(PetscLogEventEnd(PC_ApplySymmetricLeft, pc, x, y, 0));
  PetscCall(VecLockReadPop(x));
  if (pc->erroriffailure) PetscCall(VecValidValues_Internal(y, 3, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCApplySymmetricRight - Applies the right part of a symmetric preconditioner to a vector.

  Collective

  Input Parameters:
+ pc - the `PC` preconditioner context
- x  - input vector

  Output Parameter:
. y - output vector

  Level: developer

  Note:
  Currently, this routine is implemented only for `PCICC` and `PCJACOBI` preconditioners.

.seealso: [](ch_ksp), `PC`, `PCApply()`, `PCApplySymmetricLeft()`
@*/
PetscErrorCode PCApplySymmetricRight(PC pc, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 3);
  PetscCheck(x != y, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_IDN, "x and y must be different vectors");
  if (pc->erroriffailure) PetscCall(VecValidValues_Internal(x, 2, PETSC_TRUE));
  PetscCall(PCSetUp(pc));
  PetscCall(VecLockReadPush(x));
  PetscCall(PetscLogEventBegin(PC_ApplySymmetricRight, pc, x, y, 0));
  PetscUseTypeMethod(pc, applysymmetricright, x, y);
  PetscCall(PetscLogEventEnd(PC_ApplySymmetricRight, pc, x, y, 0));
  PetscCall(VecLockReadPop(x));
  if (pc->erroriffailure) PetscCall(VecValidValues_Internal(y, 3, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCApplyTranspose - Applies the transpose of preconditioner to a vector.

  Collective

  Input Parameters:
+ pc - the `PC` preconditioner context
- x  - input vector

  Output Parameter:
. y - output vector

  Level: developer

  Note:
  For complex numbers this applies the non-Hermitian transpose.

  Developer Note:
  We need to implement a `PCApplyHermitianTranspose()`

.seealso: [](ch_ksp), `PC`, `PCApply()`, `PCApplyBAorAB()`, `PCApplyBAorABTranspose()`, `PCApplyTransposeExists()`
@*/
PetscErrorCode PCApplyTranspose(PC pc, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 3);
  PetscCheck(x != y, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_IDN, "x and y must be different vectors");
  if (pc->erroriffailure) PetscCall(VecValidValues_Internal(x, 2, PETSC_TRUE));
  PetscCall(PCSetUp(pc));
  PetscCall(VecLockReadPush(x));
  PetscCall(PetscLogEventBegin(PC_Apply, pc, x, y, 0));
  PetscUseTypeMethod(pc, applytranspose, x, y);
  PetscCall(PetscLogEventEnd(PC_Apply, pc, x, y, 0));
  PetscCall(VecLockReadPop(x));
  if (pc->erroriffailure) PetscCall(VecValidValues_Internal(y, 3, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCApplyTransposeExists - Test whether the preconditioner has a transpose apply operation

  Collective

  Input Parameter:
. pc - the `PC` preconditioner context

  Output Parameter:
. flg - `PETSC_TRUE` if a transpose operation is defined

  Level: developer

.seealso: [](ch_ksp), `PC`, `PCApplyTranspose()`
@*/
PetscErrorCode PCApplyTransposeExists(PC pc, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscAssertPointer(flg, 2);
  if (pc->ops->applytranspose) *flg = PETSC_TRUE;
  else *flg = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCApplyBAorAB - Applies the preconditioner and operator to a vector. $y = B*A*x $ or $ y = A*B*x$.

  Collective

  Input Parameters:
+ pc   - the `PC` preconditioner context
. side - indicates the preconditioner side, one of `PC_LEFT`, `PC_RIGHT`, or `PC_SYMMETRIC`
. x    - input vector
- work - work vector

  Output Parameter:
. y - output vector

  Level: developer

  Note:
  If the `PC` has had `PCSetDiagonalScale()` set then $ D M A D^{-1} $ for left preconditioning or $ D A M D^{-1} $ is actually applied.
  The specific `KSPSolve()` method must also be written to handle the post-solve "correction" for the diagonal scaling.

.seealso: [](ch_ksp), `PC`, `PCApply()`, `PCApplyTranspose()`, `PCApplyBAorABTranspose()`
@*/
PetscErrorCode PCApplyBAorAB(PC pc, PCSide side, Vec x, Vec y, Vec work)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(pc, side, 2);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 4);
  PetscValidHeaderSpecific(work, VEC_CLASSID, 5);
  PetscCheckSameComm(pc, 1, x, 3);
  PetscCheckSameComm(pc, 1, y, 4);
  PetscCheckSameComm(pc, 1, work, 5);
  PetscCheck(x != y, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_IDN, "x and y must be different vectors");
  PetscCheck(side == PC_LEFT || side == PC_SYMMETRIC || side == PC_RIGHT, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_OUTOFRANGE, "Side must be right, left, or symmetric");
  PetscCheck(!pc->diagonalscale || side != PC_SYMMETRIC, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "Cannot include diagonal scaling with symmetric preconditioner application");
  if (pc->erroriffailure) PetscCall(VecValidValues_Internal(x, 3, PETSC_TRUE));

  PetscCall(PCSetUp(pc));
  if (pc->diagonalscale) {
    if (pc->ops->applyBA) {
      Vec work2; /* this is expensive, but to fix requires a second work vector argument to PCApplyBAorAB() */
      PetscCall(VecDuplicate(x, &work2));
      PetscCall(PCDiagonalScaleRight(pc, x, work2));
      PetscUseTypeMethod(pc, applyBA, side, work2, y, work);
      PetscCall(PCDiagonalScaleLeft(pc, y, y));
      PetscCall(VecDestroy(&work2));
    } else if (side == PC_RIGHT) {
      PetscCall(PCDiagonalScaleRight(pc, x, y));
      PetscCall(PCApply(pc, y, work));
      PetscCall(MatMult(pc->mat, work, y));
      PetscCall(PCDiagonalScaleLeft(pc, y, y));
    } else if (side == PC_LEFT) {
      PetscCall(PCDiagonalScaleRight(pc, x, y));
      PetscCall(MatMult(pc->mat, y, work));
      PetscCall(PCApply(pc, work, y));
      PetscCall(PCDiagonalScaleLeft(pc, y, y));
    } else PetscCheck(side != PC_SYMMETRIC, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "Cannot provide diagonal scaling with symmetric application of preconditioner");
  } else {
    if (pc->ops->applyBA) {
      PetscUseTypeMethod(pc, applyBA, side, x, y, work);
    } else if (side == PC_RIGHT) {
      PetscCall(PCApply(pc, x, work));
      PetscCall(MatMult(pc->mat, work, y));
    } else if (side == PC_LEFT) {
      PetscCall(MatMult(pc->mat, x, work));
      PetscCall(PCApply(pc, work, y));
    } else if (side == PC_SYMMETRIC) {
      /* There's an extra copy here; maybe should provide 2 work vectors instead? */
      PetscCall(PCApplySymmetricRight(pc, x, work));
      PetscCall(MatMult(pc->mat, work, y));
      PetscCall(VecCopy(y, work));
      PetscCall(PCApplySymmetricLeft(pc, work, y));
    }
  }
  if (pc->erroriffailure) PetscCall(VecValidValues_Internal(y, 4, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCApplyBAorABTranspose - Applies the transpose of the preconditioner
  and operator to a vector. That is, applies $B^T * A^T$ with left preconditioning,
  NOT $(B*A)^T = A^T*B^T$.

  Collective

  Input Parameters:
+ pc   - the `PC` preconditioner context
. side - indicates the preconditioner side, one of `PC_LEFT`, `PC_RIGHT`, or `PC_SYMMETRIC`
. x    - input vector
- work - work vector

  Output Parameter:
. y - output vector

  Level: developer

  Note:
  This routine is used internally so that the same Krylov code can be used to solve $A x = b$ and $A^T x = b$, with a preconditioner
  defined by $B^T$. This is why this has the funny form that it computes $B^T * A^T$

.seealso: [](ch_ksp), `PC`, `PCApply()`, `PCApplyTranspose()`, `PCApplyBAorAB()`
@*/
PetscErrorCode PCApplyBAorABTranspose(PC pc, PCSide side, Vec x, Vec y, Vec work)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 4);
  PetscValidHeaderSpecific(work, VEC_CLASSID, 5);
  PetscCheck(x != y, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_IDN, "x and y must be different vectors");
  if (pc->erroriffailure) PetscCall(VecValidValues_Internal(x, 3, PETSC_TRUE));
  if (pc->ops->applyBAtranspose) {
    PetscUseTypeMethod(pc, applyBAtranspose, side, x, y, work);
    if (pc->erroriffailure) PetscCall(VecValidValues_Internal(y, 4, PETSC_FALSE));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(side == PC_LEFT || side == PC_RIGHT, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_OUTOFRANGE, "Side must be right or left");

  PetscCall(PCSetUp(pc));
  if (side == PC_RIGHT) {
    PetscCall(PCApplyTranspose(pc, x, work));
    PetscCall(MatMultTranspose(pc->mat, work, y));
  } else if (side == PC_LEFT) {
    PetscCall(MatMultTranspose(pc->mat, x, work));
    PetscCall(PCApplyTranspose(pc, work, y));
  }
  /* add support for PC_SYMMETRIC */
  if (pc->erroriffailure) PetscCall(VecValidValues_Internal(y, 4, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCApplyRichardsonExists - Determines whether a particular preconditioner has a
  built-in fast application of Richardson's method.

  Not Collective

  Input Parameter:
. pc - the preconditioner

  Output Parameter:
. exists - `PETSC_TRUE` or `PETSC_FALSE`

  Level: developer

.seealso: [](ch_ksp), `PC`, `KSPRICHARDSON`, `PCApplyRichardson()`
@*/
PetscErrorCode PCApplyRichardsonExists(PC pc, PetscBool *exists)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscAssertPointer(exists, 2);
  if (pc->ops->applyrichardson) *exists = PETSC_TRUE;
  else *exists = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCApplyRichardson - Applies several steps of Richardson iteration with
  the particular preconditioner. This routine is usually used by the
  Krylov solvers and not the application code directly.

  Collective

  Input Parameters:
+ pc        - the `PC` preconditioner context
. b         - the right-hand side
. w         - one work vector
. rtol      - relative decrease in residual norm convergence criteria
. abstol    - absolute residual norm convergence criteria
. dtol      - divergence residual norm increase criteria
. its       - the number of iterations to apply.
- guesszero - if the input x contains nonzero initial guess

  Output Parameters:
+ outits - number of iterations actually used (for SOR this always equals its)
. reason - the reason the apply terminated
- y      - the solution (also contains initial guess if guesszero is `PETSC_FALSE`

  Level: developer

  Notes:
  Most preconditioners do not support this function. Use the command
  `PCApplyRichardsonExists()` to determine if one does.

  Except for the `PCMG` this routine ignores the convergence tolerances
  and always runs for the number of iterations

.seealso: [](ch_ksp), `PC`, `PCApplyRichardsonExists()`
@*/
PetscErrorCode PCApplyRichardson(PC pc, Vec b, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool guesszero, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(b, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(w, VEC_CLASSID, 4);
  PetscCheck(b != y, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_IDN, "b and y must be different vectors");
  PetscCall(PCSetUp(pc));
  PetscUseTypeMethod(pc, applyrichardson, b, y, w, rtol, abstol, dtol, its, guesszero, outits, reason);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCSetFailedReason - Sets the reason a `PCSetUp()` failed or `PC_NOERROR` if it did not fail

  Logically Collective

  Input Parameters:
+ pc     - the `PC` preconditioner context
- reason - the reason it failed

  Level: advanced

.seealso: [](ch_ksp), `PC`, `PCCreate()`, `PCApply()`, `PCDestroy()`, `PCFailedReason`
@*/
PetscErrorCode PCSetFailedReason(PC pc, PCFailedReason reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  pc->failedreason = reason;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCGetFailedReason - Gets the reason a `PCSetUp()` failed or `PC_NOERROR` if it did not fail

  Not Collective

  Input Parameter:
. pc - the `PC` preconditioner context

  Output Parameter:
. reason - the reason it failed

  Level: advanced

  Note:
  After a call to `KSPCheckDot()` or  `KSPCheckNorm()` inside a `KSPSolve()` or a call to `PCReduceFailedReason()`
  this is the maximum reason over all MPI processes in the `PC` communicator and hence logically collective.
  Otherwise it returns the local value.

.seealso: [](ch_ksp), `PC`, `PCCreate()`, `PCApply()`, `PCDestroy()`, `PCSetFailedReason()`, `PCFailedReason`
@*/
PetscErrorCode PCGetFailedReason(PC pc, PCFailedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  *reason = pc->failedreason;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCReduceFailedReason - Reduce the failed reason among the MPI processes that share the `PC`

  Collective

  Input Parameter:
. pc - the `PC` preconditioner context

  Level: advanced

  Note:
  Different MPI processes may have different reasons or no reason, see `PCGetFailedReason()`. This routine
  makes them have a common value (failure if any MPI process had a failure).

.seealso: [](ch_ksp), `PC`, `PCCreate()`, `PCApply()`, `PCDestroy()`, `PCGetFailedReason()`, `PCSetFailedReason()`, `PCFailedReason`
@*/
PetscErrorCode PCReduceFailedReason(PC pc)
{
  PetscInt buf;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  buf = (PetscInt)pc->failedreason;
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &buf, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)pc)));
  pc->failedreason = (PCFailedReason)buf;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
      a setupcall of 0 indicates never setup,
                     1 indicates has been previously setup
                    -1 indicates a PCSetUp() was attempted and failed
*/
/*@
  PCSetUp - Prepares for the use of a preconditioner. Performs all the one-time operations needed before the preconditioner
  can be used with `PCApply()`

  Collective

  Input Parameter:
. pc - the `PC` preconditioner context

  Level: developer

  Notes:
  For example, for `PCLU` this will compute the factorization.

  This is called automatically by `KSPSetUp()` or `PCApply()` so rarely needs to be called directly.

  For nested preconditioners, such as `PCFIELDSPLIT` or `PCBJACOBI` this may not finish the construction of the preconditioner
  on the inner levels, the routine `PCSetUpOnBlocks()` may compute more of the preconditioner in those situations.

.seealso: [](ch_ksp), `PC`, `PCCreate()`, `PCApply()`, `PCDestroy()`, `KSPSetUp()`, `PCSetUpOnBlocks()`
@*/
PetscErrorCode PCSetUp(PC pc)
{
  const char      *def;
  PetscObjectState matstate, matnonzerostate;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscCheck(pc->mat, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "Matrix must be set first");

  if (pc->setupcalled && pc->reusepreconditioner) {
    PetscCall(PetscInfo(pc, "Leaving PC with identical preconditioner since reuse preconditioner is set\n"));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscObjectStateGet((PetscObject)pc->pmat, &matstate));
  PetscCall(MatGetNonzeroState(pc->pmat, &matnonzerostate));
  if (!pc->setupcalled) {
    //PetscCall(PetscInfo(pc, "Setting up PC for first time\n"));
    pc->flag = DIFFERENT_NONZERO_PATTERN;
  } else if (matstate == pc->matstate) PetscFunctionReturn(PETSC_SUCCESS);
  else {
    if (matnonzerostate != pc->matnonzerostate) {
      PetscCall(PetscInfo(pc, "Setting up PC with different nonzero pattern\n"));
      pc->flag = DIFFERENT_NONZERO_PATTERN;
    } else {
      //PetscCall(PetscInfo(pc, "Setting up PC with same nonzero pattern\n"));
      pc->flag = SAME_NONZERO_PATTERN;
    }
  }
  pc->matstate        = matstate;
  pc->matnonzerostate = matnonzerostate;

  if (!((PetscObject)pc)->type_name) {
    PetscCall(PCGetDefaultType_Private(pc, &def));
    PetscCall(PCSetType(pc, def));
  }

  PetscCall(MatSetErrorIfFailure(pc->pmat, pc->erroriffailure));
  PetscCall(MatSetErrorIfFailure(pc->mat, pc->erroriffailure));
  PetscCall(PetscLogEventBegin(PC_SetUp, pc, 0, 0, 0));
  if (pc->ops->setup) {
    PetscCall(PCLogEventsDeactivatePush());
    PetscUseTypeMethod(pc, setup);
    PetscCall(PCLogEventsDeactivatePop());
  }
  PetscCall(PetscLogEventEnd(PC_SetUp, pc, 0, 0, 0));
  if (pc->postsetup) PetscCall((*pc->postsetup)(pc));
  if (!pc->setupcalled) pc->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCSetUpOnBlocks - Sets up the preconditioner for each block in
  the block Jacobi, overlapping Schwarz, and fieldsplit methods.

  Collective

  Input Parameter:
. pc - the `PC` preconditioner context

  Level: developer

  Notes:
  For nested preconditioners such as `PCBJACOBI`, `PCSetUp()` is not called on each sub-`KSP` when `PCSetUp()` is
  called on the outer `PC`, this routine ensures it is called.

  It calls `PCSetUp()` if not yet called.

.seealso: [](ch_ksp), `PC`, `PCSetUp()`, `PCCreate()`, `PCApply()`, `PCDestroy()`
@*/
PetscErrorCode PCSetUpOnBlocks(PC pc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  if (!pc->setupcalled) PetscCall(PCSetUp(pc)); /* "if" to prevent -info extra prints */
  if (!pc->ops->setuponblocks) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatSetErrorIfFailure(pc->pmat, pc->erroriffailure));
  PetscCall(PetscLogEventBegin(PC_SetUpOnBlocks, pc, 0, 0, 0));
  PetscCall(PCLogEventsDeactivatePush());
  PetscUseTypeMethod(pc, setuponblocks);
  PetscCall(PCLogEventsDeactivatePop());
  PetscCall(PetscLogEventEnd(PC_SetUpOnBlocks, pc, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCSetModifySubMatrices - Sets a user-defined routine for modifying the
  submatrices that arise within certain subdomain-based preconditioners such as `PCASM`

  Logically Collective

  Input Parameters:
+ pc   - the `PC` preconditioner context
. func - routine for modifying the submatrices, see `PCModifySubMatricesFn`
- ctx  - optional user-defined context (may be `NULL`)

  Level: advanced

  Notes:
  The basic submatrices are extracted from the matrix used to construct the preconditioner as
  usual; the user can then alter these (for example, to set different boundary
  conditions for each submatrix) before they are used for the local solves.

  `PCSetModifySubMatrices()` MUST be called before `KSPSetUp()` and
  `KSPSolve()`.

  A routine set by `PCSetModifySubMatrices()` is currently called within
  the block Jacobi (`PCBJACOBI`) and additive Schwarz (`PCASM`)
  preconditioners.  All other preconditioners ignore this routine.

.seealso: [](ch_ksp), `PC`, `PCModifySubMatricesFn`, `PCBJACOBI`, `PCASM`, `PCModifySubMatrices()`
@*/
PetscErrorCode PCSetModifySubMatrices(PC pc, PCModifySubMatricesFn *func, void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  pc->modifysubmatrices  = func;
  pc->modifysubmatricesP = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCModifySubMatrices - Calls an optional user-defined routine within
  certain preconditioners if one has been set with `PCSetModifySubMatrices()`.

  Collective

  Input Parameters:
+ pc     - the `PC` preconditioner context
. nsub   - the number of local submatrices
. row    - an array of index sets that contain the global row numbers
         that comprise each local submatrix
. col    - an array of index sets that contain the global column numbers
         that comprise each local submatrix
. submat - array of local submatrices
- ctx    - optional user-defined context for private data for the
         user-defined routine (may be `NULL`)

  Output Parameter:
. submat - array of local submatrices (the entries of which may
            have been modified)

  Level: developer

  Note:
  The user should NOT generally call this routine, as it will
  automatically be called within certain preconditioners.

.seealso: [](ch_ksp), `PC`, `PCModifySubMatricesFn`, `PCSetModifySubMatrices()`
@*/
PetscErrorCode PCModifySubMatrices(PC pc, PetscInt nsub, const IS row[], const IS col[], Mat submat[], void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  if (!pc->modifysubmatrices) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventBegin(PC_ModifySubMatrices, pc, 0, 0, 0));
  PetscCall((*pc->modifysubmatrices)(pc, nsub, row, col, submat, ctx));
  PetscCall(PetscLogEventEnd(PC_ModifySubMatrices, pc, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCSetOperators - Sets the matrix associated with the linear system and
  a (possibly) different one from which the preconditioner will be constructed.

  Logically Collective

  Input Parameters:
+ pc   - the `PC` preconditioner context
. Amat - the matrix that defines the linear system
- Pmat - the matrix to be used in constructing the preconditioner, usually the same as Amat.

  Level: advanced

  Notes:
  Using this routine directly is rarely needed, the preferred, and equivalent, usage is `KSPSetOperators()`.

  Passing a `NULL` for `Amat` or `Pmat` removes the matrix that is currently used.

  If you wish to replace either `Amat` or `Pmat` but leave the other one untouched then
  first call `KSPGetOperators()` to get the one you wish to keep, call `PetscObjectReference()`
  on it and then pass it back in in your call to `KSPSetOperators()`.

  More Notes about Repeated Solution of Linear Systems:
  PETSc does NOT reset the matrix entries of either `Amat` or `Pmat`
  to zero after a linear solve; the user is completely responsible for
  matrix assembly.  See the routine `MatZeroEntries()` if desiring to
  zero all elements of a matrix.

.seealso: [](ch_ksp), `PC`, `PCGetOperators()`, `MatZeroEntries()`
 @*/
PetscErrorCode PCSetOperators(PC pc, Mat Amat, Mat Pmat)
{
  PetscInt m1, n1, m2, n2;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  if (Amat) PetscValidHeaderSpecific(Amat, MAT_CLASSID, 2);
  if (Pmat) PetscValidHeaderSpecific(Pmat, MAT_CLASSID, 3);
  if (Amat) PetscCheckSameComm(pc, 1, Amat, 2);
  if (Pmat) PetscCheckSameComm(pc, 1, Pmat, 3);
  if (pc->setupcalled && pc->mat && pc->pmat && Amat && Pmat) {
    PetscCall(MatGetLocalSize(Amat, &m1, &n1));
    PetscCall(MatGetLocalSize(pc->mat, &m2, &n2));
    PetscCheck(m1 == m2 && n1 == n2, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Cannot change local size of Amat after use old sizes %" PetscInt_FMT " %" PetscInt_FMT " new sizes %" PetscInt_FMT " %" PetscInt_FMT, m2, n2, m1, n1);
    PetscCall(MatGetLocalSize(Pmat, &m1, &n1));
    PetscCall(MatGetLocalSize(pc->pmat, &m2, &n2));
    PetscCheck(m1 == m2 && n1 == n2, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Cannot change local size of Pmat after use old sizes %" PetscInt_FMT " %" PetscInt_FMT " new sizes %" PetscInt_FMT " %" PetscInt_FMT, m2, n2, m1, n1);
  }

  if (Pmat != pc->pmat) {
    /* changing the operator that defines the preconditioner thus reneed to clear current states so new preconditioner is built */
    pc->matnonzerostate = -1;
    pc->matstate        = -1;
  }

  /* reference first in case the matrices are the same */
  if (Amat) PetscCall(PetscObjectReference((PetscObject)Amat));
  PetscCall(MatDestroy(&pc->mat));
  if (Pmat) PetscCall(PetscObjectReference((PetscObject)Pmat));
  PetscCall(MatDestroy(&pc->pmat));
  pc->mat  = Amat;
  pc->pmat = Pmat;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCSetReusePreconditioner - reuse the current preconditioner even if the operator in the preconditioner `PC` has changed.

  Logically Collective

  Input Parameters:
+ pc   - the `PC` preconditioner context
- flag - `PETSC_TRUE` do not compute a new preconditioner, `PETSC_FALSE` do compute a new preconditioner

  Level: intermediate

  Note:
  Normally if a matrix inside a `PC` changes the `PC` automatically updates itself using information from the changed matrix. This option
  prevents this.

.seealso: [](ch_ksp), `PC`, `PCGetOperators()`, `MatZeroEntries()`, `PCGetReusePreconditioner()`, `KSPSetReusePreconditioner()`
 @*/
PetscErrorCode PCSetReusePreconditioner(PC pc, PetscBool flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidLogicalCollectiveBool(pc, flag, 2);
  pc->reusepreconditioner = flag;
  PetscTryMethod(pc, "PCSetReusePreconditioner_C", (PC, PetscBool), (pc, flag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCGetReusePreconditioner - Determines if the `PC` reuses the current preconditioner even if the operator in the preconditioner has changed.

  Not Collective

  Input Parameter:
. pc - the `PC` preconditioner context

  Output Parameter:
. flag - `PETSC_TRUE` do not compute a new preconditioner, `PETSC_FALSE` do compute a new preconditioner

  Level: intermediate

.seealso: [](ch_ksp), `PC`, `PCGetOperators()`, `MatZeroEntries()`, `PCSetReusePreconditioner()`
 @*/
PetscErrorCode PCGetReusePreconditioner(PC pc, PetscBool *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscAssertPointer(flag, 2);
  *flag = pc->reusepreconditioner;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCGetOperators - Gets the matrix associated with the linear system and
  possibly a different one which is used to construct the preconditioner.

  Not Collective, though parallel `Mat`s are returned if `pc` is parallel

  Input Parameter:
. pc - the `PC` preconditioner context

  Output Parameters:
+ Amat - the matrix defining the linear system
- Pmat - the matrix from which the preconditioner is constructed, usually the same as Amat.

  Level: intermediate

  Note:
  Does not increase the reference count of the matrices, so you should not destroy them

  Alternative usage: If the operators have NOT been set with `KSPSetOperators()`/`PCSetOperators()` then the operators
  are created in `PC` and returned to the user. In this case, if both operators
  mat and pmat are requested, two DIFFERENT operators will be returned. If
  only one is requested both operators in the PC will be the same (i.e. as
  if one had called `KSPSetOperators()`/`PCSetOperators()` with the same argument for both Mats).
  The user must set the sizes of the returned matrices and their type etc just
  as if the user created them with `MatCreate()`. For example,

.vb
         KSP/PCGetOperators(ksp/pc,&Amat,NULL); is equivalent to
           set size, type, etc of Amat

         MatCreate(comm,&mat);
         KSP/PCSetOperators(ksp/pc,Amat,Amat);
         PetscObjectDereference((PetscObject)mat);
           set size, type, etc of Amat
.ve

  and

.vb
         KSP/PCGetOperators(ksp/pc,&Amat,&Pmat); is equivalent to
           set size, type, etc of Amat and Pmat

         MatCreate(comm,&Amat);
         MatCreate(comm,&Pmat);
         KSP/PCSetOperators(ksp/pc,Amat,Pmat);
         PetscObjectDereference((PetscObject)Amat);
         PetscObjectDereference((PetscObject)Pmat);
           set size, type, etc of Amat and Pmat
.ve

  The rationale for this support is so that when creating a `TS`, `SNES`, or `KSP` the hierarchy
  of underlying objects (i.e. `SNES`, `KSP`, `PC`, `Mat`) and their lifespans can be completely
  managed by the top most level object (i.e. the `TS`, `SNES`, or `KSP`). Another way to look
  at this is when you create a `SNES` you do not NEED to create a `KSP` and attach it to
  the `SNES` object (the `SNES` object manages it for you). Similarly when you create a KSP
  you do not need to attach a `PC` to it (the `KSP` object manages the `PC` object for you).
  Thus, why should YOU have to create the `Mat` and attach it to the `SNES`/`KSP`/`PC`, when
  it can be created for you?

.seealso: [](ch_ksp), `PC`, `PCSetOperators()`, `KSPGetOperators()`, `KSPSetOperators()`, `PCGetOperatorsSet()`
@*/
PetscErrorCode PCGetOperators(PC pc, Mat *Amat, Mat *Pmat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  if (Amat) {
    if (!pc->mat) {
      if (pc->pmat && !Pmat) { /* Pmat has been set, but user did not request it, so use for Amat */
        pc->mat = pc->pmat;
        PetscCall(PetscObjectReference((PetscObject)pc->mat));
      } else { /* both Amat and Pmat are empty */
        PetscCall(MatCreate(PetscObjectComm((PetscObject)pc), &pc->mat));
        if (!Pmat) { /* user did NOT request Pmat, so make same as Amat */
          pc->pmat = pc->mat;
          PetscCall(PetscObjectReference((PetscObject)pc->pmat));
        }
      }
    }
    *Amat = pc->mat;
  }
  if (Pmat) {
    if (!pc->pmat) {
      if (pc->mat && !Amat) { /* Amat has been set but was not requested, so use for pmat */
        pc->pmat = pc->mat;
        PetscCall(PetscObjectReference((PetscObject)pc->pmat));
      } else {
        PetscCall(MatCreate(PetscObjectComm((PetscObject)pc), &pc->pmat));
        if (!Amat) { /* user did NOT request Amat, so make same as Pmat */
          pc->mat = pc->pmat;
          PetscCall(PetscObjectReference((PetscObject)pc->mat));
        }
      }
    }
    *Pmat = pc->pmat;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCGetOperatorsSet - Determines if the matrix associated with the linear system and
  possibly a different one associated with the preconditioner have been set in the `PC`.

  Not Collective, though the results on all processes should be the same

  Input Parameter:
. pc - the `PC` preconditioner context

  Output Parameters:
+ mat  - the matrix associated with the linear system was set
- pmat - matrix associated with the preconditioner was set, usually the same

  Level: intermediate

.seealso: [](ch_ksp), `PC`, `PCSetOperators()`, `KSPGetOperators()`, `KSPSetOperators()`, `PCGetOperators()`
@*/
PetscErrorCode PCGetOperatorsSet(PC pc, PetscBool *mat, PetscBool *pmat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  if (mat) *mat = (pc->mat) ? PETSC_TRUE : PETSC_FALSE;
  if (pmat) *pmat = (pc->pmat) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCFactorGetMatrix - Gets the factored matrix from the
  preconditioner context.  This routine is valid only for the `PCLU`,
  `PCILU`, `PCCHOLESKY`, and `PCICC` methods.

  Not Collective though `mat` is parallel if `pc` is parallel

  Input Parameter:
. pc - the `PC` preconditioner context

  Output Parameters:
. mat - the factored matrix

  Level: advanced

  Note:
  Does not increase the reference count for `mat` so DO NOT destroy it

.seealso: [](ch_ksp), `PC`, `PCLU`, `PCILU`, `PCCHOLESKY`, `PCICC`
@*/
PetscErrorCode PCFactorGetMatrix(PC pc, Mat *mat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscAssertPointer(mat, 2);
  PetscCall(PCFactorSetUpMatSolverType(pc));
  PetscUseTypeMethod(pc, getfactoredmatrix, mat);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCSetOptionsPrefix - Sets the prefix used for searching for all
  `PC` options in the database.

  Logically Collective

  Input Parameters:
+ pc     - the `PC` preconditioner context
- prefix - the prefix string to prepend to all `PC` option requests

  Note:
  A hyphen (-) must NOT be given at the beginning of the prefix name.
  The first character of all runtime options is AUTOMATICALLY the
  hyphen.

  Level: advanced

.seealso: [](ch_ksp), `PC`, `PCSetFromOptions`, `PCAppendOptionsPrefix()`, `PCGetOptionsPrefix()`
@*/
PetscErrorCode PCSetOptionsPrefix(PC pc, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)pc, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCAppendOptionsPrefix - Appends to the prefix used for searching for all
  `PC` options in the database.

  Logically Collective

  Input Parameters:
+ pc     - the `PC` preconditioner context
- prefix - the prefix string to prepend to all `PC` option requests

  Note:
  A hyphen (-) must NOT be given at the beginning of the prefix name.
  The first character of all runtime options is AUTOMATICALLY the
  hyphen.

  Level: advanced

.seealso: [](ch_ksp), `PC`, `PCSetFromOptions`, `PCSetOptionsPrefix()`, `PCGetOptionsPrefix()`
@*/
PetscErrorCode PCAppendOptionsPrefix(PC pc, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)pc, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCGetOptionsPrefix - Gets the prefix used for searching for all
  PC options in the database.

  Not Collective

  Input Parameter:
. pc - the `PC` preconditioner context

  Output Parameter:
. prefix - pointer to the prefix string used, is returned

  Level: advanced

.seealso: [](ch_ksp), `PC`, `PCSetFromOptions`, `PCSetOptionsPrefix()`, `PCAppendOptionsPrefix()`
@*/
PetscErrorCode PCGetOptionsPrefix(PC pc, const char *prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscAssertPointer(prefix, 2);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)pc, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Indicates the right-hand side will be changed by KSPSolve(), this occurs for a few
  preconditioners including BDDC and Eisentat that transform the equations before applying
  the Krylov methods
*/
PETSC_INTERN PetscErrorCode PCPreSolveChangeRHS(PC pc, PetscBool *change)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscAssertPointer(change, 2);
  *change = PETSC_FALSE;
  PetscTryMethod(pc, "PCPreSolveChangeRHS_C", (PC, PetscBool *), (pc, change));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCPreSolve - Optional pre-solve phase, intended for any preconditioner-specific actions that must be performed before
  the iterative solve itself. Used in conjunction with `PCPostSolve()`

  Collective

  Input Parameters:
+ pc  - the `PC` preconditioner context
- ksp - the Krylov subspace context

  Level: developer

  Notes:
  `KSPSolve()` calls this directly, so is rarely called by the user.

  Certain preconditioners, such as the `PCType` of `PCEISENSTAT`, change the formulation of the linear system to be solved iteratively.
  This function performs that transformation. `PCPostSolve()` then transforms the system back to its original form after the solve.
  `PCPostSolve()` also transforms the resulting solution of the transformed system to the solution of the original problem.

  `KSPSetPostSolve()` provides an alternative way to provide such transformations.

.seealso: [](ch_ksp), `PC`, `PCPostSolve()`, `KSP`, `PCSetPostSetUp()`, `KSPSetPreSolve()`, `KSPSetPostSolve()`
@*/
PetscErrorCode PCPreSolve(PC pc, KSP ksp)
{
  Vec x, rhs;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 2);
  pc->presolvedone++;
  PetscCheck(pc->presolvedone <= 2, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "Cannot embed PCPreSolve() more than twice");
  PetscCall(KSPGetSolution(ksp, &x));
  PetscCall(KSPGetRhs(ksp, &rhs));
  PetscTryTypeMethod(pc, presolve, ksp, rhs, x);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCSetPostSetUp - Sets function called at the end of `PCSetUp()` to adjust the computed preconditioner

  Logically Collective

  Input Parameters:
+ pc        - the preconditioner object
- postsetup - the function to call after `PCSetUp()`

  Calling sequence of `postsetup`:
. pc - the `PC` context

  Level: developer

.seealso: [](ch_ksp), `PC`, `PCSetUp()`
@*/
PetscErrorCode PCSetPostSetUp(PC pc, PetscErrorCode (*postsetup)(PC pc))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  pc->postsetup = postsetup;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCPostSolve - Optional post-solve phase, intended for any
  preconditioner-specific actions that must be performed after
  the iterative solve itself.

  Collective

  Input Parameters:
+ pc  - the `PC` preconditioner context
- ksp - the `KSP` Krylov subspace context

  Example Usage:
.vb
    PCPreSolve(pc,ksp);
    KSPSolve(ksp,b,x);
    PCPostSolve(pc,ksp);
.ve

  Level: developer

  Note:
  `KSPSolve()` calls this routine directly, so it is rarely called by the user.

.seealso: [](ch_ksp), `PC`, `KSPSetPostSolve()`, `KSPSetPreSolve()`, `PCPreSolve()`, `KSPSolve()`
@*/
PetscErrorCode PCPostSolve(PC pc, KSP ksp)
{
  Vec x, rhs;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 2);
  pc->presolvedone--;
  PetscCall(KSPGetSolution(ksp, &x));
  PetscCall(KSPGetRhs(ksp, &rhs));
  PetscTryTypeMethod(pc, postsolve, ksp, rhs, x);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCLoad - Loads a `PC` that has been stored in binary  with `PCView()`.

  Collective

  Input Parameters:
+ newdm  - the newly loaded `PC`, this needs to have been created with `PCCreate()` or
           some related function before a call to `PCLoad()`.
- viewer - binary file viewer `PETSCVIEWERBINARY`, obtained from `PetscViewerBinaryOpen()`

  Level: intermediate

  Note:
  The type is determined by the data in the file, any `PCType` set into the `PC` before this call is ignored.

.seealso: [](ch_ksp), `PC`, `PetscViewerBinaryOpen()`, `PCView()`, `MatLoad()`, `VecLoad()`, `PETSCVIEWERBINARY`
@*/
PetscErrorCode PCLoad(PC newdm, PetscViewer viewer)
{
  PetscBool isbinary;
  PetscInt  classid;
  char      type[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(newdm, PC_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
  PetscCheck(isbinary, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid viewer; open viewer with PetscViewerBinaryOpen()");

  PetscCall(PetscViewerBinaryRead(viewer, &classid, 1, NULL, PETSC_INT));
  PetscCheck(classid == PC_FILE_CLASSID, PetscObjectComm((PetscObject)newdm), PETSC_ERR_ARG_WRONG, "Not PC next in file");
  PetscCall(PetscViewerBinaryRead(viewer, type, 256, NULL, PETSC_CHAR));
  PetscCall(PCSetType(newdm, type));
  PetscTryTypeMethod(newdm, load, viewer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petscdraw.h>
#if defined(PETSC_HAVE_SAWS)
  #include <petscviewersaws.h>
#endif

/*@
  PCViewFromOptions - View (print or provide information about) the `PC`, based on options in the options database

  Collective

  Input Parameters:
+ A    - the `PC` context
. obj  - Optional object that provides the options prefix
- name - command line option name

  Level: developer

.seealso: [](ch_ksp), `PC`, `PCView`, `PetscObjectViewFromOptions()`, `PCCreate()`
@*/
PetscErrorCode PCViewFromOptions(PC A, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, PC_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCView - Prints information about the `PC`

  Collective

  Input Parameters:
+ pc     - the `PC` preconditioner context
- viewer - optional `PetscViewer` visualization context

  Level: intermediate

  Notes:
  The available visualization contexts include
+     `PETSC_VIEWER_STDOUT_SELF` - standard output (default)
-     `PETSC_VIEWER_STDOUT_WORLD` - synchronized standard
  output where only the first processor opens
  the file. All other processors send their
  data to the first processor to print.

  The user can open an alternative visualization contexts with
  `PetscViewerASCIIOpen()` (output to a specified file).

.seealso: [](ch_ksp), `PC`, `PetscViewer`, `PetscViewerType`, `KSPView()`, `PetscViewerASCIIOpen()`
@*/
PetscErrorCode PCView(PC pc, PetscViewer viewer)
{
  PCType            cstr;
  PetscViewerFormat format;
  PetscBool         isascii, isstring, isbinary, isdraw, pop = PETSC_FALSE;
#if defined(PETSC_HAVE_SAWS)
  PetscBool issaws;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)pc), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(pc, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSTRING, &isstring));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERDRAW, &isdraw));
#if defined(PETSC_HAVE_SAWS)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSAWS, &issaws));
#endif

  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)pc, viewer));
    if (!pc->setupcalled) PetscCall(PetscViewerASCIIPrintf(viewer, "  PC has not been set up so information may be incomplete\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscTryTypeMethod(pc, view, viewer);
    PetscCall(PetscViewerASCIIPopTab(viewer));
    if (pc->mat) {
      PetscCall(PetscViewerGetFormat(viewer, &format));
      if (format != PETSC_VIEWER_ASCII_INFO_DETAIL) {
        PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO));
        pop = PETSC_TRUE;
      }
      if (pc->pmat == pc->mat) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "  linear system matrix = precond matrix:\n"));
        PetscCall(PetscViewerASCIIPushTab(viewer));
        PetscCall(MatView(pc->mat, viewer));
        PetscCall(PetscViewerASCIIPopTab(viewer));
      } else {
        if (pc->pmat) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "  linear system matrix followed by preconditioner matrix:\n"));
        } else {
          PetscCall(PetscViewerASCIIPrintf(viewer, "  linear system matrix:\n"));
        }
        PetscCall(PetscViewerASCIIPushTab(viewer));
        PetscCall(MatView(pc->mat, viewer));
        if (pc->pmat) PetscCall(MatView(pc->pmat, viewer));
        PetscCall(PetscViewerASCIIPopTab(viewer));
      }
      if (pop) PetscCall(PetscViewerPopFormat(viewer));
    }
  } else if (isstring) {
    PetscCall(PCGetType(pc, &cstr));
    PetscCall(PetscViewerStringSPrintf(viewer, " PCType: %-7.7s", cstr));
    PetscTryTypeMethod(pc, view, viewer);
    if (pc->mat) PetscCall(MatView(pc->mat, viewer));
    if (pc->pmat && pc->pmat != pc->mat) PetscCall(MatView(pc->pmat, viewer));
  } else if (isbinary) {
    PetscInt    classid = PC_FILE_CLASSID;
    MPI_Comm    comm;
    PetscMPIInt rank;
    char        type[256];

    PetscCall(PetscObjectGetComm((PetscObject)pc, &comm));
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    if (rank == 0) {
      PetscCall(PetscViewerBinaryWrite(viewer, &classid, 1, PETSC_INT));
      PetscCall(PetscStrncpy(type, ((PetscObject)pc)->type_name, 256));
      PetscCall(PetscViewerBinaryWrite(viewer, type, 256, PETSC_CHAR));
    }
    PetscTryTypeMethod(pc, view, viewer);
  } else if (isdraw) {
    PetscDraw draw;
    char      str[25];
    PetscReal x, y, bottom, h;
    PetscInt  n;

    PetscCall(PetscViewerDrawGetDraw(viewer, 0, &draw));
    PetscCall(PetscDrawGetCurrentPoint(draw, &x, &y));
    if (pc->mat) {
      PetscCall(MatGetSize(pc->mat, &n, NULL));
      PetscCall(PetscSNPrintf(str, 25, "PC: %s (%" PetscInt_FMT ")", ((PetscObject)pc)->type_name, n));
    } else {
      PetscCall(PetscSNPrintf(str, 25, "PC: %s", ((PetscObject)pc)->type_name));
    }
    PetscCall(PetscDrawStringBoxed(draw, x, y, PETSC_DRAW_RED, PETSC_DRAW_BLACK, str, NULL, &h));
    bottom = y - h;
    PetscCall(PetscDrawPushCurrentPoint(draw, x, bottom));
    PetscTryTypeMethod(pc, view, viewer);
    PetscCall(PetscDrawPopCurrentPoint(draw));
#if defined(PETSC_HAVE_SAWS)
  } else if (issaws) {
    PetscMPIInt rank;

    PetscCall(PetscObjectName((PetscObject)pc));
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    if (!((PetscObject)pc)->amsmem && rank == 0) PetscCall(PetscObjectViewSAWs((PetscObject)pc, viewer));
    if (pc->mat) PetscCall(MatView(pc->mat, viewer));
    if (pc->pmat && pc->pmat != pc->mat) PetscCall(MatView(pc->pmat, viewer));
#endif
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCRegister -  Adds a method (`PCType`) to the PETSc preconditioner package.

  Not collective. No Fortran Support

  Input Parameters:
+ sname    - name of a new user-defined solver
- function - routine to create the method context which will be stored in a `PC` when `PCSetType()` is called

  Example Usage:
.vb
   PCRegister("my_solver", MySolverCreate);
.ve

  Then, your solver can be chosen with the procedural interface via
.vb
  PCSetType(pc, "my_solver")
.ve
  or at runtime via the option
.vb
  -pc_type my_solver
.ve

  Level: advanced

  Note:
  A simpler alternative to using `PCRegister()` for an application specific preconditioner is to use a `PC` of `PCType` `PCSHELL` and
  provide your customizations with `PCShellSetContext()` and `PCShellSetApply()`

  `PCRegister()` may be called multiple times to add several user-defined preconditioners.

.seealso: [](ch_ksp), `PC`, `PCType`, `PCRegisterAll()`, `PCSetType()`, `PCShellSetContext()`, `PCShellSetApply()`, `PCSHELL`
@*/
PetscErrorCode PCRegister(const char sname[], PetscErrorCode (*function)(PC))
{
  PetscFunctionBegin;
  PetscCall(PCInitializePackage());
  PetscCall(PetscFunctionListAdd(&PCList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_PC(Mat A, Vec X, Vec Y)
{
  PC pc;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &pc));
  PetscCall(PCApply(pc, X, Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCComputeOperator - Computes the explicit preconditioned operator as a matrix `Mat`.

  Collective

  Input Parameters:
+ pc      - the `PC` preconditioner object
- mattype - the `MatType` to be used for the operator

  Output Parameter:
. mat - the explicit preconditioned operator

  Level: advanced

  Note:
  This computation is done by applying the operators to columns of the identity matrix.
  This routine is costly in general, and is recommended for use only with relatively small systems.
  Currently, this routine uses a dense matrix format when `mattype` == `NULL`

  Developer Note:
  This should be called `PCCreateExplicitOperator()`

.seealso: [](ch_ksp), `PC`, `KSPComputeOperator()`, `MatType`
@*/
PetscErrorCode PCComputeOperator(PC pc, MatType mattype, Mat *mat)
{
  PetscInt N, M, m, n;
  Mat      A, Apc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscAssertPointer(mat, 3);
  PetscCall(PCGetOperators(pc, &A, NULL));
  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatGetSize(A, &M, &N));
  PetscCall(MatCreateShell(PetscObjectComm((PetscObject)pc), m, n, M, N, pc, &Apc));
  PetscCall(MatShellSetOperation(Apc, MATOP_MULT, (PetscErrorCodeFn *)MatMult_PC));
  PetscCall(MatComputeOperator(Apc, mattype, mat));
  PetscCall(MatDestroy(&Apc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCSetCoordinates - sets the coordinates of all the nodes (degrees of freedom in the vector) on the local process

  Collective

  Input Parameters:
+ pc     - the `PC` preconditioner context
. dim    - the dimension of the coordinates 1, 2, or 3
. nloc   - the blocked size of the coordinates array
- coords - the coordinates array

  Level: intermediate

  Notes:
  `coords` is an array of the dim coordinates for the nodes on
  the local processor, of size `dim`*`nloc`.
  If there are 108 equations (dofs) on a processor
  for a 3d displacement finite element discretization of elasticity (so
  that there are nloc = 36 = 108/3 nodes) then the array must have 108
  double precision values (ie, 3 * 36).  These x y z coordinates
  should be ordered for nodes 0 to N-1 like so: [ 0.x, 0.y, 0.z, 1.x,
  ... , N-1.z ].

  The information provided here can be used by some preconditioners, such as `PCGAMG`, to produce a better preconditioner.
  See also  `MatSetNearNullSpace()`.

.seealso: [](ch_ksp), `PC`, `MatSetNearNullSpace()`
@*/
PetscErrorCode PCSetCoordinates(PC pc, PetscInt dim, PetscInt nloc, PetscReal coords[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidLogicalCollectiveInt(pc, dim, 2);
  PetscTryMethod(pc, "PCSetCoordinates_C", (PC, PetscInt, PetscInt, PetscReal[]), (pc, dim, nloc, coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCGetInterpolations - Gets interpolation matrices for all levels (except level 0)

  Logically Collective

  Input Parameter:
. pc - the precondition context

  Output Parameters:
+ num_levels     - the number of levels
- interpolations - the interpolation matrices (size of `num_levels`-1)

  Level: advanced

  Developer Note:
  Why is this here instead of in `PCMG` etc?

.seealso: [](ch_ksp), `PC`, `PCMG`, `PCMGGetRestriction()`, `PCMGSetInterpolation()`, `PCMGGetInterpolation()`, `PCGetCoarseOperators()`
@*/
PetscErrorCode PCGetInterpolations(PC pc, PetscInt *num_levels, Mat *interpolations[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscAssertPointer(num_levels, 2);
  PetscAssertPointer(interpolations, 3);
  PetscUseMethod(pc, "PCGetInterpolations_C", (PC, PetscInt *, Mat *[]), (pc, num_levels, interpolations));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCGetCoarseOperators - Gets coarse operator matrices for all levels (except the finest level)

  Logically Collective

  Input Parameter:
. pc - the precondition context

  Output Parameters:
+ num_levels      - the number of levels
- coarseOperators - the coarse operator matrices (size of `num_levels`-1)

  Level: advanced

  Developer Note:
  Why is this here instead of in `PCMG` etc?

.seealso: [](ch_ksp), `PC`, `PCMG`, `PCMGGetRestriction()`, `PCMGSetInterpolation()`, `PCMGGetRScale()`, `PCMGGetInterpolation()`, `PCGetInterpolations()`
@*/
PetscErrorCode PCGetCoarseOperators(PC pc, PetscInt *num_levels, Mat *coarseOperators[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscAssertPointer(num_levels, 2);
  PetscAssertPointer(coarseOperators, 3);
  PetscUseMethod(pc, "PCGetCoarseOperators_C", (PC, PetscInt *, Mat *[]), (pc, num_levels, coarseOperators));
  PetscFunctionReturn(PETSC_SUCCESS);
}
