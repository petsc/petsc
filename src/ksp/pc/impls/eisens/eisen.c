
/*
   Defines a  Eisenstat trick SSOR  preconditioner. This uses about
 %50 of the usual amount of floating point ops used for SSOR + Krylov
 method. But it requires actually solving the preconditioned problem
 with both left and right preconditioning.
*/
#include <petsc/private/pcimpl.h> /*I "petscpc.h" I*/

typedef struct {
  Mat       shell, A;
  Vec       b[2], diag; /* temporary storage for true right hand side */
  PetscReal omega;
  PetscBool usediag; /* indicates preconditioner should include diagonal scaling*/
} PC_Eisenstat;

static PetscErrorCode PCMult_Eisenstat(Mat mat, Vec b, Vec x)
{
  PC            pc;
  PC_Eisenstat *eis;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat, &pc));
  eis = (PC_Eisenstat *)pc->data;
  PetscCall(MatSOR(eis->A, b, eis->omega, SOR_EISENSTAT, 0.0, 1, 1, x));
  PetscCall(MatFactorGetError(eis->A, (MatFactorError *)&pc->failedreason));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCNorm_Eisenstat(Mat mat, NormType type, PetscReal *nrm)
{
  PC            pc;
  PC_Eisenstat *eis;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat, &pc));
  eis = (PC_Eisenstat *)pc->data;
  PetscCall(MatNorm(eis->A, type, nrm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_Eisenstat(PC pc, Vec x, Vec y)
{
  PC_Eisenstat *eis = (PC_Eisenstat *)pc->data;
  PetscBool     hasop;

  PetscFunctionBegin;
  if (eis->usediag) {
    PetscCall(MatHasOperation(pc->pmat, MATOP_MULT_DIAGONAL_BLOCK, &hasop));
    if (hasop) {
      PetscCall(MatMultDiagonalBlock(pc->pmat, x, y));
    } else {
      PetscCall(VecPointwiseMult(y, x, eis->diag));
    }
  } else PetscCall(VecCopy(x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyTranspose_Eisenstat(PC pc, Vec x, Vec y)
{
  PC_Eisenstat *eis = (PC_Eisenstat *)pc->data;
  PetscBool     hasop, set, sym;

  PetscFunctionBegin;
  PetscCall(MatIsSymmetricKnown(eis->A, &set, &sym));
  PetscCheck(set && sym, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "Can only apply transpose of Eisenstat if matrix is symmetric");
  if (eis->usediag) {
    PetscCall(MatHasOperation(pc->pmat, MATOP_MULT_DIAGONAL_BLOCK, &hasop));
    if (hasop) {
      PetscCall(MatMultDiagonalBlock(pc->pmat, x, y));
    } else {
      PetscCall(VecPointwiseMult(y, x, eis->diag));
    }
  } else PetscCall(VecCopy(x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPreSolve_Eisenstat(PC pc, KSP ksp, Vec b, Vec x)
{
  PC_Eisenstat *eis = (PC_Eisenstat *)pc->data;
  PetscBool     nonzero;

  PetscFunctionBegin;
  if (pc->presolvedone < 2) {
    PetscCheck(pc->mat == pc->pmat, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "Cannot have different mat and pmat");
    /* swap shell matrix and true matrix */
    eis->A  = pc->mat;
    pc->mat = eis->shell;
  }

  if (!eis->b[pc->presolvedone - 1]) { PetscCall(VecDuplicate(b, &eis->b[pc->presolvedone - 1])); }

  /* if nonzero initial guess, modify x */
  PetscCall(KSPGetInitialGuessNonzero(ksp, &nonzero));
  if (nonzero) {
    PetscCall(VecCopy(x, eis->b[pc->presolvedone - 1]));
    PetscCall(MatSOR(eis->A, eis->b[pc->presolvedone - 1], eis->omega, SOR_APPLY_UPPER, 0.0, 1, 1, x));
    PetscCall(MatFactorGetError(eis->A, (MatFactorError *)&pc->failedreason));
  }

  /* save true b, other option is to swap pointers */
  PetscCall(VecCopy(b, eis->b[pc->presolvedone - 1]));

  /* modify b by (L + D/omega)^{-1} */
  PetscCall(MatSOR(eis->A, eis->b[pc->presolvedone - 1], eis->omega, (MatSORType)(SOR_ZERO_INITIAL_GUESS | SOR_LOCAL_FORWARD_SWEEP), 0.0, 1, 1, b));
  PetscCall(MatFactorGetError(eis->A, (MatFactorError *)&pc->failedreason));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPostSolve_Eisenstat(PC pc, KSP ksp, Vec b, Vec x)
{
  PC_Eisenstat *eis = (PC_Eisenstat *)pc->data;

  PetscFunctionBegin;
  /* get back true b */
  PetscCall(VecCopy(eis->b[pc->presolvedone], b));

  /* modify x by (U + D/omega)^{-1} */
  PetscCall(VecCopy(x, eis->b[pc->presolvedone]));
  PetscCall(MatSOR(eis->A, eis->b[pc->presolvedone], eis->omega, (MatSORType)(SOR_ZERO_INITIAL_GUESS | SOR_LOCAL_BACKWARD_SWEEP), 0.0, 1, 1, x));
  PetscCall(MatFactorGetError(eis->A, (MatFactorError *)&pc->failedreason));
  if (!pc->presolvedone) pc->mat = eis->A;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCReset_Eisenstat(PC pc)
{
  PC_Eisenstat *eis = (PC_Eisenstat *)pc->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&eis->b[0]));
  PetscCall(VecDestroy(&eis->b[1]));
  PetscCall(MatDestroy(&eis->shell));
  PetscCall(VecDestroy(&eis->diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_Eisenstat(PC pc)
{
  PetscFunctionBegin;
  PetscCall(PCReset_Eisenstat(pc));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCEisenstatSetOmega_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCEisenstatSetNoDiagonalScaling_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCEisenstatGetOmega_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCEisenstatGetNoDiagonalScaling_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPreSolveChangeRHS_C", NULL));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_Eisenstat(PC pc, PetscOptionItems *PetscOptionsObject)
{
  PC_Eisenstat *eis = (PC_Eisenstat *)pc->data;
  PetscBool     set, flg;
  PetscReal     omega;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Eisenstat SSOR options");
  PetscCall(PetscOptionsReal("-pc_eisenstat_omega", "Relaxation factor 0 < omega < 2", "PCEisenstatSetOmega", eis->omega, &omega, &flg));
  if (flg) PetscCall(PCEisenstatSetOmega(pc, omega));
  PetscCall(PetscOptionsBool("-pc_eisenstat_no_diagonal_scaling", "Do not use standard diagonal scaling", "PCEisenstatSetNoDiagonalScaling", eis->usediag ? PETSC_FALSE : PETSC_TRUE, &flg, &set));
  if (set) PetscCall(PCEisenstatSetNoDiagonalScaling(pc, flg));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_Eisenstat(PC pc, PetscViewer viewer)
{
  PC_Eisenstat *eis = (PC_Eisenstat *)pc->data;
  PetscBool     iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  omega = %g\n", (double)eis->omega));
    if (eis->usediag) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Using diagonal scaling (default)\n"));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Not using diagonal scaling\n"));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_Eisenstat(PC pc)
{
  PetscInt      M, N, m, n;
  PetscBool     set, sym;
  PC_Eisenstat *eis = (PC_Eisenstat *)pc->data;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    PetscCall(MatGetSize(pc->mat, &M, &N));
    PetscCall(MatGetLocalSize(pc->mat, &m, &n));
    PetscCall(MatIsSymmetricKnown(pc->mat, &set, &sym));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)pc), &eis->shell));
    PetscCall(MatSetSizes(eis->shell, m, n, M, N));
    PetscCall(MatSetType(eis->shell, MATSHELL));
    PetscCall(MatSetUp(eis->shell));
    PetscCall(MatShellSetContext(eis->shell, pc));
    PetscCall(MatShellSetOperation(eis->shell, MATOP_MULT, (void (*)(void))PCMult_Eisenstat));
    if (set && sym) PetscCall(MatShellSetOperation(eis->shell, MATOP_MULT_TRANSPOSE, (void (*)(void))PCMult_Eisenstat));
    PetscCall(MatShellSetOperation(eis->shell, MATOP_NORM, (void (*)(void))PCNorm_Eisenstat));
  }
  if (!eis->usediag) PetscFunctionReturn(PETSC_SUCCESS);
  if (!pc->setupcalled) { PetscCall(MatCreateVecs(pc->pmat, &eis->diag, NULL)); }
  PetscCall(MatGetDiagonal(pc->pmat, eis->diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --------------------------------------------------------------------*/

static PetscErrorCode PCEisenstatSetOmega_Eisenstat(PC pc, PetscReal omega)
{
  PC_Eisenstat *eis = (PC_Eisenstat *)pc->data;

  PetscFunctionBegin;
  PetscCheck(omega > 0.0 && omega < 2.0, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_OUTOFRANGE, "Relaxation out of range");
  eis->omega = omega;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCEisenstatSetNoDiagonalScaling_Eisenstat(PC pc, PetscBool flg)
{
  PC_Eisenstat *eis = (PC_Eisenstat *)pc->data;

  PetscFunctionBegin;
  eis->usediag = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCEisenstatGetOmega_Eisenstat(PC pc, PetscReal *omega)
{
  PC_Eisenstat *eis = (PC_Eisenstat *)pc->data;

  PetscFunctionBegin;
  *omega = eis->omega;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCEisenstatGetNoDiagonalScaling_Eisenstat(PC pc, PetscBool *flg)
{
  PC_Eisenstat *eis = (PC_Eisenstat *)pc->data;

  PetscFunctionBegin;
  *flg = eis->usediag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PCEisenstatSetOmega - Sets the SSOR relaxation coefficient, omega,
   to use with Eisenstat's trick (where omega = 1.0 by default)

   Logically Collective

   Input Parameters:
+  pc - the preconditioner context
-  omega - relaxation coefficient (0 < omega < 2)

   Options Database Key:
.  -pc_eisenstat_omega <omega> - Sets omega

   Level: intermediate

   Notes:
   The Eisenstat trick implementation of SSOR requires about 50% of the
   usual amount of floating point operations used for SSOR + Krylov method;
   however, the preconditioned problem must be solved with both left
   and right preconditioning.

   To use SSOR without the Eisenstat trick, employ the `PCSOR` preconditioner,
   which can be chosen with the database options
$    -pc_type  sor  -pc_sor_symmetric

.seealso: `PCSORSetOmega()`, `PCEISENSTAT`
@*/
PetscErrorCode PCEisenstatSetOmega(PC pc, PetscReal omega)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidLogicalCollectiveReal(pc, omega, 2);
  PetscTryMethod(pc, "PCEisenstatSetOmega_C", (PC, PetscReal), (pc, omega));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PCEisenstatSetNoDiagonalScaling - Causes the Eisenstat preconditioner, `PCEISENSTAT`
   not to do additional diagonal preconditioning. For matrices with a constant
   along the diagonal, this may save a small amount of work.

   Logically Collective

   Input Parameters:
+  pc - the preconditioner context
-  flg - `PETSC_TRUE` turns off diagonal scaling inside the algorithm

   Options Database Key:
.  -pc_eisenstat_no_diagonal_scaling - Activates `PCEisenstatSetNoDiagonalScaling()`

   Level: intermediate

   Note:
     If you use the `KSPSetDiagonalScaling()` or -ksp_diagonal_scale option then you will
   likely want to use this routine since it will save you some unneeded flops.

.seealso: `PCEisenstatSetOmega()`, `PCEISENSTAT`
@*/
PetscErrorCode PCEisenstatSetNoDiagonalScaling(PC pc, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCEisenstatSetNoDiagonalScaling_C", (PC, PetscBool), (pc, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PCEisenstatGetOmega - Gets the SSOR relaxation coefficient, omega,
   to use with Eisenstat's trick (where omega = 1.0 by default).

   Logically Collective

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  omega - relaxation coefficient (0 < omega < 2)

   Options Database Key:
.  -pc_eisenstat_omega <omega> - Sets omega

   Notes:
   The Eisenstat trick implementation of SSOR requires about 50% of the
   usual amount of floating point operations used for SSOR + Krylov method;
   however, the preconditioned problem must be solved with both left
   and right preconditioning.

   To use SSOR without the Eisenstat trick, employ the PCSOR preconditioner,
   which can be chosen with the database options
$    -pc_type  sor  -pc_sor_symmetric

   Level: intermediate

.seealso: `PCEISENSTAT`, `PCSORGetOmega()`, `PCEisenstatSetOmega()`
@*/
PetscErrorCode PCEisenstatGetOmega(PC pc, PetscReal *omega)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscUseMethod(pc, "PCEisenstatGetOmega_C", (PC, PetscReal *), (pc, omega));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PCEisenstatGetNoDiagonalScaling - Tells if the Eisenstat preconditioner
   not to do additional diagonal preconditioning. For matrices with a constant
   along the diagonal, this may save a small amount of work.

   Logically Collective

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  flg - `PETSC_TRUE` means there is no diagonal scaling applied

   Options Database Key:
.  -pc_eisenstat_no_diagonal_scaling - Activates `PCEisenstatSetNoDiagonalScaling()`

   Level: intermediate

   Note:
     If you use the KSPSetDiagonalScaling() or -ksp_diagonal_scale option then you will
   likely want to use this routine since it will save you some unneeded flops.

.seealso: , `PCEISENSTAT`, `PCEisenstatGetOmega()`
@*/
PetscErrorCode PCEisenstatGetNoDiagonalScaling(PC pc, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscUseMethod(pc, "PCEisenstatGetNoDiagonalScaling_C", (PC, PetscBool *), (pc, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPreSolveChangeRHS_Eisenstat(PC pc, PetscBool *change)
{
  PetscFunctionBegin;
  *change = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     PCEISENSTAT - An implementation of SSOR (symmetric successive over relaxation, symmetric Gauss-Seidel)
           preconditioning that incorporates Eisenstat's trick to reduce the amount of computation needed.

   Options Database Keys:
+  -pc_eisenstat_omega <omega> - Sets omega
-  -pc_eisenstat_no_diagonal_scaling - Activates `PCEisenstatSetNoDiagonalScaling()`

   Level: beginner

   Notes:
   Only implemented for the `MATAIJ` matrix format.

   Not a true parallel SOR, in parallel this implementation corresponds to block Jacobi with SOR on each block.

   Developer Note:
   Since this algorithm runs the Krylov method on a transformed linear system the implementation provides `PCPreSolve()` and `PCPostSolve()`
   routines that `KSP` uses to set up the transformed linear system.

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PC`, `PCEisenstatGetOmega()`,
          `PCEisenstatSetNoDiagonalScaling()`, `PCEisenstatSetOmega()`, `PCSOR`
M*/

PETSC_EXTERN PetscErrorCode PCCreate_Eisenstat(PC pc)
{
  PC_Eisenstat *eis;

  PetscFunctionBegin;
  PetscCall(PetscNew(&eis));

  pc->ops->apply           = PCApply_Eisenstat;
  pc->ops->applytranspose  = PCApplyTranspose_Eisenstat;
  pc->ops->presolve        = PCPreSolve_Eisenstat;
  pc->ops->postsolve       = PCPostSolve_Eisenstat;
  pc->ops->applyrichardson = NULL;
  pc->ops->setfromoptions  = PCSetFromOptions_Eisenstat;
  pc->ops->destroy         = PCDestroy_Eisenstat;
  pc->ops->reset           = PCReset_Eisenstat;
  pc->ops->view            = PCView_Eisenstat;
  pc->ops->setup           = PCSetUp_Eisenstat;

  pc->data     = eis;
  eis->omega   = 1.0;
  eis->b[0]    = NULL;
  eis->b[1]    = NULL;
  eis->diag    = NULL;
  eis->usediag = PETSC_TRUE;

  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCEisenstatSetOmega_C", PCEisenstatSetOmega_Eisenstat));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCEisenstatSetNoDiagonalScaling_C", PCEisenstatSetNoDiagonalScaling_Eisenstat));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCEisenstatGetOmega_C", PCEisenstatGetOmega_Eisenstat));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCEisenstatGetNoDiagonalScaling_C", PCEisenstatGetNoDiagonalScaling_Eisenstat));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPreSolveChangeRHS_C", PCPreSolveChangeRHS_Eisenstat));
  PetscFunctionReturn(PETSC_SUCCESS);
}
